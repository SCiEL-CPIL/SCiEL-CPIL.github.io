import time
import random
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)  # 448

    # 1. Generate edges and map them to indices
    edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx

    # 2. Generate faces (4-cycles)
    faces = []
    face_set = set()

    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j

            for x in range(n_vertices):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    v0 = x
                    v1 = x ^ mask_i
                    v2 = x ^ mask_j
                    v3 = x ^ mask_i ^ mask_j

                    e01 = edge_to_idx[(min(v0, v1), max(v0, v1))]
                    e13 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                    e32 = edge_to_idx[(min(v3, v2), max(v3, v2))]
                    e20 = edge_to_idx[(min(v2, v0), max(v2, v0))]

                    face_edges = (e01, e13, e32, e20)
                    face_key = tuple(sorted(face_edges))

                    if face_key not in face_set:
                        face_set.add(face_key)
                        faces.append(face_key)

    n_faces = len(faces)

    # 3. Precompute edge-to-face incidence
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)

    def greedy_solve(order, current_time=None):
        state = [False] * n_edges
        face_counts = [0] * n_faces

        for e_idx in order:
            valid = True
            for f_idx in edge_faces[e_idx]:
                if face_counts[f_idx] >= 3:
                    valid = False
                    break
            if valid:
                state[e_idx] = True
                for f_idx in edge_faces[e_idx]:
                    face_counts[f_idx] += 1

        return state, face_counts

    def local_search(state, face_counts, deadline):
        current_count = sum(state)
        best_state = state[:]
        best_count = current_count
        best_face_counts = face_counts[:]

        all_edges = list(range(n_edges))

        while time.time() < deadline:
            improved = False

            # Strategy 1: Try to add a single edge
            for _ in range(50):
                e_idx = random.choice(all_edges)
                if state[e_idx]:
                    continue

                valid = True
                for f_idx in edge_faces[e_idx]:
                    if face_counts[f_idx] >= 3:
                        valid = False
                        break

                if valid:
                    state[e_idx] = True
                    for f_idx in edge_faces[e_idx]:
                        face_counts[f_idx] += 1
                    current_count += 1
                    improved = True
                    break

            if improved:
                if current_count > best_count:
                    best_count = current_count
                    best_state = state[:]
                    best_face_counts = face_counts[:]
                continue

            # Strategy 2: Remove one edge, then try to add as many as possible
            e_out = None
            for _ in range(100):
                candidate = random.choice(all_edges)
                if state[candidate]:
                    e_out = candidate
                    break

            if e_out is None:
                break

            state[e_out] = False
            for f_idx in edge_faces[e_out]:
                face_counts[f_idx] -= 1
            current_count -= 1

            candidates = set()
            for f_idx in edge_faces[e_out]:
                for e_in_face in faces[f_idx]:
                    if not state[e_in_face]:
                        candidates.add(e_in_face)

            cand_list = list(candidates)
            random.shuffle(cand_list)

            added_any = False
            for e_in in cand_list:
                valid = True
                for f_idx in edge_faces[e_in]:
                    if face_counts[f_idx] >= 3:
                        valid = False
                        break

                if valid:
                    state[e_in] = True
                    for f_idx in edge_faces[e_in]:
                        face_counts[f_idx] += 1
                    current_count += 1
                    added_any = True

            if current_count >= best_count:
                if current_count > best_count:
                    best_count = current_count
                    best_state = state[:]
                    best_face_counts = face_counts[:]
            else:
                state[e_out] = True
                for f_idx in edge_faces[e_out]:
                    face_counts[f_idx] += 1
                current_count += 1
                for e_in in cand_list:
                    if state[e_in]:
                        state[e_in] = False
                        for f_idx in edge_faces[e_in]:
                            face_counts[f_idx] -= 1
                        current_count -= 1

        return best_state, best_count

    # Spectral relaxation approach
    def spectral_init():
        # Build the line graph adjacency: edges e1, e2 are adjacent if they share a face
        # Actually, for the cut formulation, we want to minimize conflicts
        # Let's build a matrix where A[e1,e2] = 1 if e1 and e2 are in the same face
        # Then we want to maximize sum x_e * x_f for non-adjacent, minimize for adjacent
        # This is equivalent to maximizing x^T (J - A_line) x where A_line has 1s for face-sharing edges
        
        # Simpler: Build the constraint matrix. 
        # For each face, we have constraint sum_{e in face} x_e <= 3.
        # Relaxation: Minimize sum_{faces} (sum_{e in face} x_e - 3)^2 + lambda * sum x_e^2
        # Or use the "spin" model: x_e in {-1, 1}. 
        # If we select an edge, x_e = 1, else -1.
        # We want to minimize the number of "bad" faces (faces with 4 selected).
        # A face is bad if all 4 are selected.
        # Product of spins in a face: P_f = prod_{e in f} x_e.
        # If all 4 are 1, P_f = 1. If not all 1, P_f = -1 (since at least one is -1).
        # Wait, if 3 are 1 and 1 is -1, P_f = -1. If 2 are 1 and 2 are -1, P_f = 1.
        # So minimizing sum P_f doesn't directly work.
        
        # Let's stick to the greedy + local search which is proven to work well, 
        # but use the spectral vector to guide the initial greedy order.
        
        # Construct the face-edge incidence matrix M (n_faces x n_edges)
        # M[f, e] = 1 if edge e is in face f
        rows = []
        cols = []
        vals = []
        for f_idx, face in enumerate(faces):
            for e_idx in face:
                rows.append(f_idx)
                cols.append(e_idx)
                vals.append(1.0)
        
        M = coo_matrix((vals, (rows, cols)), shape=(n_faces, n_edges)).tocsr()
        
        # We want to find a vector v such that M v is close to a vector of 2s (since avg degree in face is 2 if we select 2/4 edges)
        # Or simply use the eigenvector of the graph Laplacian of the hypergraph?
        
        # Let's try to maximize the number of edges subject to M x <= 3, x in {0,1}.
        # Relax to x in [0,1]. 
        # Use a simple gradient ascent on the relaxed objective:
        # Maximize sum x_e - penalty * sum_f max(0, sum_{e in f} x_e - 3)^2
        
        x = np.ones(n_edges) * 0.5
        lr = 0.01
        penalty = 10.0
        
        # Precompute M^T for gradient
        MT = M.T.tocsr()
        
        # Gradient of sum x is 1
        # Gradient of penalty term: 
        # g_e = - 2 * penalty * sum_{f: e in f} max(0, sum_{e' in f} x_e' - 3)
        
        for step in range(500):
            if time.time() > deadline - 10:
                break
            face_sums = M.dot(x)
            violations = np.maximum(0, face_sums - 3)
            
            # Gradient of penalty
            grad_penalty = MT.dot(2 * penalty * violations)
            
            # Total gradient
            grad = np.ones(n_edges) - grad_penalty
            
            x = x + lr * grad
            x = np.clip(x, 0, 1)
        
        # Sort edges by x value descending for greedy initialization
        order = list(np.argsort(-x))
        return order

    # Main Loop: Multiple Restarts
    best_overall_count = 0
    best_overall_state = None

    # Run several restarts
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            # First restart uses spectral initialization
            order = spectral_init()
        else:
            # Subsequent restarts use random order
            order = list(range(n_edges))
            random.shuffle(order)

        state, face_counts = greedy_solve(order)
        count = sum(state)

        if count > best_overall_count:
            best_overall_count = count
            best_overall_state = state[:]

        # Local search
        if time.time() < deadline:
            improved_state, improved_count = local_search(state, face_counts, deadline)
            if improved_count > best_overall_count:
                best_overall_count = improved_count
                best_overall_state = improved_state[:]

        # Check time
        if time.time() > deadline:
            break

    # Convert best state to list of edges
    result = []
    if best_overall_state:
        for i, is_selected in enumerate(best_overall_state):
            if is_selected:
                u, v = edges[i]
                result.append([u, v])
    else:
        # Fallback: Empty or simple greedy
        order = list(range(n_edges))
        state, _ = greedy_solve(order)
        for i, is_selected in enumerate(state):
            if is_selected:
                u, v = edges[i]
                result.append([u, v])

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
