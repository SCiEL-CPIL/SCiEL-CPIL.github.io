import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1)) # 7 * 64 = 448
    
    # 1. Precompute Edge-Index Mapping
    # Map (u, v) to a unique index 0..447
    # We can encode an edge by (min(u,v), bit_index)
    # Or simpler: iterate u, then i, if u < u^i, assign index.
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    assert idx == num_edges
    
    # 2. Precompute Face Constraints
    # A 2D face is defined by two distinct bit positions i, j and a base vertex u 
    # where bits i and j are 0 in u (to make it unique per face).
    # Actually, simpler: iterate all pairs of bits (i,j) and all base vertices u 
    # such that bit i and bit j are 0 in u.
    # The 4 edges of the face are:
    # e1: (u, u^i)
    # e2: (u, u^j)
    # e3: (u^i, u^i^j)
    # e4: (u^j, u^i^j)
    
    # We want to represent the constraint: NOT (e1 AND e2 AND e3 AND e4)
    # For fast checking in NumPy, we can store for each edge e, the list of 
    # "cliques" (faces) it belongs to. 
    # However, checking if adding edge e completes a face requires knowing 
    # if the other 3 edges in any face containing e are already selected.
    
    # Alternative representation for speed:
    # For each edge e, store the list of faces it is part of.
    # For each face, store the 4 edge indices.
    
    faces = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all u such that bit i and bit j are 0
            # There are 2^(d-2) such u's
            # We can generate them by iterating all u and filtering, or constructing directly.
            # Filtering is fine since n=128.
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    # Get indices
                    ie1 = edge_to_idx[e1]
                    ie2 = edge_to_idx[e2]
                    ie3 = edge_to_idx[e3]
                    ie4 = edge_to_idx[e4]
                    
                    face_indices = (ie1, ie2, ie3, ie4)
                    faces.append(face_indices)
                    
                    for ie in face_indices:
                        edge_to_faces[ie].append(len(faces) - 1)
                        
    num_faces = len(faces)
    # Convert faces to a numpy array for vectorized operations if needed, 
    # but for local search, list of tuples is often faster for sparse checks.
    
    # 3. Local Search Strategy
    # We will run multiple independent local searches (hill climbing) 
    # starting from random valid configurations.
    # To generate a random valid config:
    #   Shuffle edges, add if valid. This is O(E * F) per edge, which is slow.
    #   Better: Start with empty set, add random edges that don't violate constraints.
    #   Since we start empty, adding is always safe until we hit a constraint.
    #   We can just do a greedy random construction.
    
    def greedy_random_construct(seed):
        random.seed(seed)
        edges = list(range(num_edges))
        random.shuffle(edges)
        selected = np.zeros(num_edges, dtype=bool)
        
        # To check validity of adding edge e:
        # Check all faces containing e. If any face has 3 other edges selected, reject.
        for e in edges:
            valid = True
            for f_idx in edge_to_faces[e]:
                f_edges = faces[f_idx]
                # Count selected edges in this face
                # We need to check the 3 other edges
                count = 0
                for oe in f_edges:
                    if oe != e and selected[oe]:
                        count += 1
                if count == 3:
                    valid = False
                    break
            if valid:
                selected[e] = True
                
        return selected

    best_count = 0
    best_selected = np.zeros(num_edges, dtype=bool)
    
    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline:
            break
            
        # Greedy construction
        current_selected = greedy_random_construct(seed)
        seed += 1
        current_count = np.sum(current_selected)
        
        # Local Search: Random flips
        # We try to add edges (improvement) or remove edges (if stuck, but we want max, so mostly add).
        # Since we start with a maximal greedy set (no edge can be added without violating a constraint? 
        # No, greedy random doesn't guarantee maximality in the sense of no single add possible? 
        # Actually, if we scan all edges and add if valid, at the end, NO edge can be added 
        # because we checked every edge. So it is a "maximal" independent set in the 
        # 4-uniform hypergraph of faces. 
        # Thus, we MUST remove edges to allow adding others to potentially increase total count?
        # Wait. If we have a maximal set, removing one edge frees up its faces. 
        # Adding it back is the only option if we want to keep it. 
        # To improve count, we must remove k edges and add k+1 edges.
        # Simple 1-flip local search (remove 1, add 1) cannot improve count 
        # because we started with a maximal set (no single add is possible).
        # So we need to allow "sideways" moves or use Simulated Annealing.
        
        # Let's use a simple SA loop for each restart.
        T = 10.0
        cooling = 0.999
        steps = 0
        max_steps = 2000 # Limit per restart to allow many restarts
        
        while steps < max_steps and time.time() < deadline:
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            
            if current_selected[e]:
                # Try to remove
                # Removing is always valid.
                # But we only want to remove if it helps us add more later?
                # In SA, we accept removals with prob exp(-delta/T). delta = -1.
                # So prob = exp(-1/T).
                if random.random() < np.exp(-1.0 / T):
                    current_selected[e] = False
                    current_count -= 1
                # else reject
            else:
                # Try to add
                # Check validity
                valid = True
                for f_idx in edge_to_faces[e]:
                    f_edges = faces[f_idx]
                    count = 0
                    for oe in f_edges:
                        if oe != e and current_selected[oe]:
                            count += 1
                    if count == 3:
                        valid = False
                        break
                
                if valid:
                    # delta = +1, always accept
                    current_selected[e] = True
                    current_count += 1
                else:
                    # Cannot add
                    pass
            
            if current_count > best_count:
                best_count = current_count
                best_selected = current_selected.copy()
                
            T *= cooling
            if T < 0.01:
                T = 0.01
            steps += 1
            
            # Occasionally check time
            if steps % 100 == 0:
                if time.time() > deadline:
                    break

    # Convert best_selected to list of edges
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result

# Optional: Quick test to ensure it runs
if __name__ == "__main__":
    res = construct()
    print(len(res))
