import numpy as np
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
    
    # Build face_to_edges and edge_to_faces
    face_to_edges = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ mask_ij
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    idx4 = edge_to_idx[e4]
                    face_idx = len(face_to_edges)
                    face_to_edges.append([idx1, idx2, idx3, idx4])
                    edge_to_faces[idx1].append(face_idx)
                    edge_to_faces[idx2].append(face_idx)
                    edge_to_faces[idx3].append(face_idx)
                    edge_to_faces[idx4].append(face_idx)
    
    n_faces = len(face_to_edges)
    
    # Max-2SAT local search:
    # Variables: x[e] = 1 if edge e is selected
    # Objective: maximize sum(x[e]) - lambda * sum(violated_faces)
    # A face is violated if all 4 edges are selected
    # We use a smooth surrogate: penalty = lambda * (sum of x in face) > 3
    
    # Simulated annealing with precise delta updates
    def simulated_annealing(max_time=135.0):
        # Initialize: select all edges, then fix violations
        x = np.ones(n_edges, dtype=np.int8)  # 1 = selected
        face_count = np.zeros(n_faces, dtype=np.int8)
        
        # Compute face counts
        for e in range(n_edges):
            for f in edge_to_faces[e]:
                face_count[f] += 1
        
        # Fix violations: remove edges from violated faces
        for f in range(n_faces):
            if face_count[f] == 4:
                # Remove a random edge from this face
                e = random.choice(face_to_edges[f])
                x[e] = 0
                for f2 in edge_to_faces[e]:
                    face_count[f2] -= 1
        
        # Objective function: edge_count - penalty * violated_count
        def objective(x, face_count):
            edge_count = np.sum(x)
            violated = np.sum(face_count == 4)
            return edge_count - 100 * violated  # Heavy penalty for violations
        
        current_obj = objective(x, face_count)
        best_x = x.copy()
        best_face_count = face_count.copy()
        best_obj = current_obj
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < max_time:
            T *= cooling
            if T < T_min:
                break
            
            # Random edge flip
            e = random.randint(0, n_edges - 1)
            
            if x[e] == 1:
                # Try unselecting
                delta_edge = -1
                delta_penalty = 0
                for f in edge_to_faces[e]:
                    if face_count[f] == 4:
                        delta_penalty += 1  # Removing this edge fixes a violation
                delta_obj = delta_edge + 100 * delta_penalty
            else:
                # Try selecting
                delta_edge = 1
                delta_penalty = 0
                for f in edge_to_faces[e]:
                    if face_count[f] == 3:
                        delta_penalty += 1  # Adding this edge creates a violation
                delta_obj = delta_edge - 100 * delta_penalty
            
            # Accept if improves or with probability exp(delta/T)
            if delta_obj >= 0:
                accept = True
            else:
                accept = random.random() < np.exp(delta_obj / T)
            
            if accept:
                x[e] = 1 - x[e]
                for f in edge_to_faces[e]:
                    face_count[f] += (1 if x[e] == 1 else -1)
                
                current_obj += delta_obj
                
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_x = x.copy()
                    best_face_count = face_count.copy()
        
        return best_x, best_face_count
    
    # Fast greedy hitting set for initial solution
    def greedy_hitting_set():
        removed = np.zeros(n_edges, dtype=bool)
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        unhit_faces = set(range(n_faces))
        
        while unhit_faces and time.time() < deadline - 5:
            best_edge = -1
            best_hits = 0
            for e in range(n_edges):
                if removed[e]:
                    continue
                hits = 0
                for f in edge_to_faces[e]:
                    if f in unhit_faces:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_edge = e
                    if best_hits >= 4:
                        break
            
            if best_edge == -1:
                break
            
            removed[best_edge] = True
            for f in edge_to_faces[best_edge]:
                face_hit_count[f] += 1
                if face_hit_count[f] == 1:
                    unhit_faces.discard(f)
        
        return removed, face_hit_count
    
    # Local search with un-removals
    def local_search_hitting_set(removed, face_hit_count, max_time=135.0):
        removed = removed.copy()
        face_hit_count = face_hit_count.copy()
        
        improved = True
        while improved and time.time() < max_time - 5:
            improved = False
            for e in range(n_edges):
                if removed[e]:
                    can_unremove = True
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] <= 1:
                            can_unremove = False
                            break
                    if can_unremove:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                        improved = True
        
        return removed, face_hit_count
    
    # Run multiple attempts
    best_selected = None
    best_edge_count = 0
    best_violations = 999
    
    # First: greedy + local search
    removed, face_hit_count = greedy_hitting_set()
    removed, face_hit_count = local_search_hitting_set(removed, face_hit_count, max_time=deadline - 60)
    selected = ~removed
    edge_count = np.sum(selected)
    # Check for violations
    fc = np.zeros(n_faces, dtype=np.int8)
    for e in range(n_edges):
        if selected[e]:
            for f in edge_to_faces[e]:
                fc[f] += 1
    violations = np.sum(fc == 4)
    
    if violations == 0:
        best_edge_count = edge_count
        best_selected = selected.copy()
        best_violations = 0
    
    # Simulated annealing attempts
    n_sa_attempts = 5
    for attempt in range(n_sa_attempts):
        if time.time() > deadline - 20:
            break
        
        sa_x, sa_face_count = simulated_annealing(max_time=deadline - 25 - attempt * 3)
        
        # Verify and fix
        violations = np.sum(sa_face_count == 4)
        if violations > 0:
            # Fix violations
            for f in range(n_faces):
                if sa_face_count[f] == 4:
                    e = random.choice(face_to_edges[f])
                    sa_x[e] = 0
                    for f2 in edge_to_faces[e]:
                        sa_face_count[f2] -= 1
            violations = np.sum(sa_face_count == 4)
        
        edge_count = np.sum(sa_x)
        
        if violations == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_selected = sa_x.copy()
            best_violations = 0
    
    # Final polish: local search on best solution
    if best_selected is not None:
        removed = ~best_selected
        fc = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if best_selected[e]:
                for f in edge_to_faces[e]:
                    fc[f] += 1
        
        removed, fc = local_search_hitting_set(removed, fc, max_time=deadline - 5)
        selected = ~removed
        edge_count = np.sum(selected)
        
        # Verify no violations
        fc2 = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if selected[e]:
                for f in edge_to_faces[e]:
                    fc2[f] += 1
        
        if np.sum(fc2 == 4) == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_selected = selected
    
    # Convert to result
    result = []
    for e in range(n_edges):
        if best_selected[e]:
            u, v = idx_to_edge[e]
            result.append([u, v])
    
    return result
