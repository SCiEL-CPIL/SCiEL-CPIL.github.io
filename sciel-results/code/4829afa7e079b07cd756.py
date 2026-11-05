"""Combined: MCS-based ordering + face-count local search with spectral initialization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute faces (4-cycles)
    faces = []
    edge_faces = [[] for _ in range(m)]
    
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v3, v4), max(v3, v4))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v4), max(v2, v4))
                    
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    idx4 = edge_to_idx[e4]
                    
                    face_edges = (idx1, idx2, idx3, idx4)
                    faces.append(face_edges)
                    
                    for e_idx in face_edges:
                        edge_faces[e_idx].append(len(faces) - 1)
    
    num_faces = len(faces)
    
    # Build conflict graph: two edges conflict if they are in the same face
    # (i.e., they share a 2-face and are not parallel)
    # Actually for C4-freeness: a face has 4 edges. We can't select all 4.
    # The constraint is: for each face, at most 3 of its 4 edges can be selected.
    # This is not a standard independent set problem.
    # But we can think of it as: we want to maximize selected edges such that no face has all 4.
    
    # For MCS heuristic, we need a conflict structure.
    # Two edges "conflict" in the sense that if both are selected, they contribute to a potential C4.
    # Let's define: edge a and edge b are in a conflict relation if they share a face.
    # Then the problem is: find a large subset with no face fully contained.
    
    # Let's build the adjacency of the conflict graph (edges sharing a face)
    conflict_adj = [[] for _ in range(m)]
    for f_idx in range(num_faces):
        fe = faces[f_idx]
        for a in range(4):
            for b in range(a+1, 4):
                if fe[a] not in conflict_adj[fe[b]]:
                    conflict_adj[fe[a]].append(fe[b])
                    conflict_adj[fe[b]].append(fe[a])
    
    def greedy_polish(chosen, face_counts):
        """Add all possible edges."""
        improved = True
        count = int(chosen.sum())
        while improved:
            improved = False
            for idx in range(m):
                if not chosen[idx]:
                    can_add = True
                    for f_idx in edge_faces[idx]:
                        if face_counts[f_idx] >= 3:
                            can_add = False
                            break
                    if can_add:
                        chosen[idx] = True
                        count += 1
                        for f_idx in edge_faces[idx]:
                            face_counts[f_idx] += 1
                        improved = True
        return count
    
    def local_search(chosen, face_counts, count, T_init=1.0, T_min=0.001, cooling=0.9999, max_time=None):
        if max_time is None:
            max_time = deadline - time.time() - 5
        
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
        no_improve = 0
        max_no_improve = 30000
        
        T = T_init
        while time.time() < max_time and T > T_min:
            idx = random.randint(0, m - 1)
            
            if not chosen[idx]:
                # Try to add
                can_add = True
                for f_idx in edge_faces[idx]:
                    if face_counts[f_idx] >= 3:
                        can_add = False
                        break
                
                if can_add:
                    chosen[idx] = True
                    count += 1
                    for f_idx in edge_faces[idx]:
                        face_counts[f_idx] += 1
                    
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        best_face_counts = face_counts.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Try Swap: Remove a conflicting edge, add idx
                    conflict_faces = [f_idx for f_idx in edge_faces[idx] if face_counts[f_idx] == 3]
                    
                    if conflict_faces:
                        f_idx = random.choice(conflict_faces)
                        face_e = faces[f_idx]
                        candidates = [e for e in face_e if e != idx and chosen[e]]
                        if candidates:
                            rem_idx = random.choice(candidates)
                            # Remove rem_idx
                            for f_idx2 in edge_faces[rem_idx]:
                                face_counts[f_idx2] -= 1
                            chosen[rem_idx] = False
                            count -= 1
                            
                            # Check if idx can be added
                            can_add2 = True
                            for f_idx3 in edge_faces[idx]:
                                if face_counts[f_idx3] >= 3:
                                    can_add2 = False
                                    break
                            
                            if can_add2:
                                chosen[idx] = True
                                count += 1
                                for f_idx4 in edge_faces[idx]:
                                    face_counts[f_idx4] += 1
                                no_improve += 1
                            else:
                                # Revert
                                chosen[rem_idx] = True
                                count += 1
                                for f_idx2 in edge_faces[rem_idx]:
                                    face_counts[f_idx2] += 1
                                no_improve += 1
                    else:
                        no_improve += 1
            else:
                # Consider removing idx with SA probability
                if random.random() < np.exp(-1.0 / T):
                    chosen[idx] = False
                    count -= 1
                    for f_idx in edge_faces[idx]:
                        face_counts[f_idx] -= 1
                    no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                chosen = best_chosen.copy()
                face_counts = best_face_counts.copy()
                count = best_count
                T = max(T, 1.0)
                no_improve = 0
        
        return best_chosen, best_face_counts, best_count
    
    def mcs_ordering():
        """Maximum Cardinality Search to get an ordering of edges.
        Returns a list of edge indices in the order they were 'discovered'."""
        # MCS for maximum independent set heuristic:
        # We want to order edges so that we process "least constrained" first.
        # In MCS for coloring, we process most constrained first.
        # For independent set, we might want to process edges with fewer conflicts first?
        # Actually, let's use a different approach: 
        # Greedy independent set on the conflict graph won't work directly 
        # because our constraint is "no face has all 4 edges", not "no two adjacent in conflict graph".
        
        # Let's use a weighted greedy: at each step, pick the edge that maximizes 
        # (number of faces it's in that are still "open") - (penalty for faces that become closed)
        # Or simply: use the spectral initialization as a priority, then greedily add.
        
        # For now, let's implement a simple MCS-like ordering:
        # Start with all edges unvisited.
        # Score of an edge = number of faces containing it that have < 3 selected edges.
        # Pick highest score, mark as selected, update scores.
        
        chosen = np.zeros(m, dtype=bool)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        remaining = set(range(m))
        
        # Initial scores: how many faces each edge is in (all start at 0 count)
        # All faces have 0 selected, so all edges have all their faces "open"
        
        order = []
        while remaining:
            # Pick edge with maximum number of "open" faces (faces with count < 3)
            best_idx = -1
            best_score = -1
            for idx in remaining:
                score = 0
                for f_idx in edge_faces[idx]:
                    if face_counts[f_idx] < 3:
                        score += 1
                # Tie-break: prefer edges with fewer total conflicts? Or random
                if score > best_score or (score == best_score and (best_idx == -1 or random.random() < 0.5)):
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            order.append(best_idx)
            remaining.discard(best_idx)
            
            # Tentatively select this edge
            # Check if it can be added
            can_add = True
            for f_idx in edge_faces[best_idx]:
                if face_counts[f_idx] >= 3:
                    can_add = False
                    break
            
            if can_add:
                chosen[best_idx] = True
                for f_idx in edge_faces[best_idx]:
                    face_counts[f_idx] += 1
        
        return chosen, face_counts, int(chosen.sum())
    
    def spectral_init():
        """Use Fiedler vector of conflict graph for initial selection."""
        # Build sparse adjacency
        row = []
        col = []
        data = []
        for i in range(m):
            for j in conflict_adj[i]:
                if i < j:
                    row.append(i)
                    col.append(j)
                    data.append(1.0)
                    row.append(j)
                    col.append(i)
                    data.append(1.0)
        
        if len(data) == 0:
            return np.random.randn(m)
        
        A = csr_matrix((data, (row, col)), shape=(m, m))
        L = laplacian(A, normed=True)
        
        try:
            eigenvalues, eigenvectors = eigsh(L, k=3, sigma=0, which='LM')
            idx_sort = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx_sort[1]]
        except Exception:
            fiedler = np.random.randn(m)
            fiedler -= fiedler.mean()
        
        return fiedler
    
    # Main search
    best_count = 0
    best_chosen = None
    best_face_counts = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: MCS-based greedy
    chosen, face_counts, count = mcs_ordering()
    count = greedy_polish(chosen, face_counts)
    
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Strategy 2: Spectral initialization + local search
    fiedler = spectral_init()
    chosen = np.zeros(m, dtype=bool)
    face_counts = np.zeros(num_faces, dtype=np.int32)
    
    # Sort edges by fiedler value (descending) and greedily add
    edge_order = np.argsort(-fiedler)
    for idx in edge_order:
        idx = int(idx)
        can_add = True
        for f_idx in edge_faces[idx]:
            if face_counts[f_idx] >= 3:
                can_add = False
                break
        if can_add:
            chosen[idx] = True
            for f_idx in edge_faces[idx]:
                face_counts[f_idx] += 1
    
    count = int(chosen.sum())
    count = greedy_polish(chosen, face_counts)
    
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Local search from best
    remaining = deadline - time.time() - 10
    if remaining > 5:
        chosen, face_counts, count = local_search(best_chosen, best_face_counts, best_count, 
                                                   T_init=1.0, T_min=0.001, cooling=0.9999, max_time=remaining)
        count = greedy_polish(chosen, face_counts)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
    
    # Multiple restarts with different strategies
    for r in range(8):
        if time.time() > deadline - 15:
            break
        
        random.seed(100 + r * 7)
        np.random.seed(100 + r * 7)
        
        if r % 2 == 0:
            # MCS with random tie-breaking
            chosen, face_counts, count = mcs_ordering()
        else:
            # Spectral with perturbation
            fiedler = spectral_init()
            fiedler += np.random.randn(m) * 0.1
            chosen = np.zeros(m, dtype=bool)
            face_counts = np.zeros(num_faces, dtype=np.int32)
            edge_order = np.argsort(-fiedler)
            for idx in edge_order:
                idx = int(idx)
                can_add = True
                for f_idx in edge_faces[idx]:
                    if face_counts[f_idx] >= 3:
                        can_add = False
                        break
                if can_add:
                    chosen[idx] = True
                    for f_idx in edge_faces[idx]:
                        face_counts[f_idx] += 1
            count = int(chosen.sum())
        
        count = greedy_polish(chosen, face_counts)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 10:
            chosen, face_counts, count = local_search(chosen, face_counts, count,
                                                       T_init=0.5, T_min=0.001, cooling=0.9999, max_time=remaining)
            count = greedy_polish(chosen, face_counts)
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
                best_face_counts = face_counts.copy()
    
    # Final polish
    count = greedy_polish(best_chosen, best_face_counts)
    if count > best_count:
        best_count = count
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
