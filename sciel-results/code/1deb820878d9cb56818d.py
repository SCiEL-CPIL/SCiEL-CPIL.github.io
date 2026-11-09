"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to leave 10s safety margin
    
    d = 7
    n = 2 ** d
    
    # Generate all edges
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Precompute for each edge, which faces it belongs to
    # A face is defined by two dimensions i, j and a vertex x (with bits i,j fixed to x's values)
    # Edge (x, x^(1<<i)) is in faces with dimensions (i, j) for j != i
    # For face (i, j), the four vertices are x, x^(1<<i), x^(1<<j), x^(1<<(i)|1<<(j))
    # where x has bit i=0 (wlog)
    
    # Better: for each edge, precompute the list of faces it participates in
    # A face is uniquely identified by (min_dim, max_dim, base_vertex) where base_vertex has both dim bits = 0
    edge_to_faces = {}
    for idx, (x, y) in enumerate(all_edges):
        # Find which bit differs
        diff = x ^ y
        i = diff.bit_length() - 1
        faces = []
        for j in range(d):
            if j == i:
                continue
            # The face has dimensions i and j
            # Base vertex: clear both bits i and j from x
            base = x & ~(1 << i) & ~(1 << j)
            mi, mj = min(i, j), max(i, j)
            faces.append((mi, mj, base))
        edge_to_faces[idx] = faces
    
    # For each face, list the edge indices that form it
    face_to_edges = {}
    for idx, (x, y) in enumerate(all_edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            base = x & ~(1 << i) & ~(1 << j)
            mi, mj = min(i, j), max(i, j)
            key = (mi, mj, base)
            if key not in face_to_edges:
                face_to_edges[key] = set()
            face_to_edges[key].add(idx)
    
    # Verify each face has exactly 4 edges
    for key, edges in face_to_edges.items():
        assert len(edges) == 4, f"Face {key} has {len(edges)} edges"
    
    # Greedy initial solution
    chosen = set()
    for idx, (x, y) in enumerate(all_edges):
        ok = True
        for face in edge_to_faces[idx]:
            if len(face_to_edges[face] & chosen) == 3:
                ok = False
                break
        if ok:
            chosen.add(idx)
    
    def is_valid(chosen_set):
        for face, edges in face_to_edges.items():
            if len(edges & chosen_set) == 4:
                return False
        return True
    
    def count_edges(chosen_set):
        return len(chosen_set)
    
    best_set = set(chosen)
    best_count = count_edges(best_set)
    
    # Local search: try to improve by removing an edge from a bad face and adding others
    # Or: try swapping - remove one edge, add another
    
    def try_add_edge(chosen_set, idx):
        """Check if edge idx can be added without creating a bad face"""
        for face in edge_to_faces[idx]:
            if len(face_to_edges[face] & chosen_set) == 3:
                return False
        return True
    
    def greedy_add(chosen_set):
        """Try to add as many edges as possible"""
        changed = True
        while changed:
            changed = False
            for idx in range(len(all_edges)):
                if idx not in chosen_set and try_add_edge(chosen_set, idx):
                    chosen_set.add(idx)
                    changed = True
                    if time.time() > deadline:
                        return chosen_set
        return chosen_set
    
    # Phase 1: Greedy add to initial solution
    chosen = greedy_add(chosen)
    if count_edges(chosen) > best_count:
        best_set = set(chosen)
        best_count = count_edges(best_set)
    
    # Phase 2: Local search - try removing one edge and adding multiple
    # For each edge in chosen, try removing it and see if we can add more
    random.seed(42)
    
    # Build list of edges for random access
    edge_list = list(range(len(all_edges)))
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try a random removal + greedy add
        if best_set:
            # Remove a random edge from best_set
            remove_candidates = list(best_set)
            random.shuffle(remove_candidates)
            
            for rem_idx in remove_candidates[:3]:  # Try up to 3 removals
                new_set = set(best_set)
                new_set.discard(rem_idx)
                new_set = greedy_add(new_set)
                if count_edges(new_set) > best_count:
                    best_set = new_set
                    best_count = count_edges(best_set)
                    break
    
    # Phase 3: Try random restarts with different greedy orders
    restarts = 10
    for r in range(restarts):
        if time.time() > deadline:
            break
        
        random.shuffle(edge_list)
        chosen_r = set()
        for idx in edge_list:
            if try_add_edge(chosen_r, idx):
                chosen_r.add(idx)
        chosen_r = greedy_add(chosen_r)
        
        if count_edges(chosen_r) > best_count:
            best_set = set(chosen_r)
            best_count = count_edges(best_set)
        
        # Local search on this new solution
        for it2 in range(500):
            if time.time() > deadline:
                break
            if best_set:
                remove_candidates = list(best_set)
                random.shuffle(remove_candidates)
                for rem_idx in remove_candidates[:2]:
                    new_set = set(best_set)
                    new_set.discard(rem_idx)
                    new_set = greedy_add(new_set)
                    if count_edges(new_set) > best_count:
                        best_set = new_set
                        best_count = count_edges(best_set)
                        break
    
    # Convert to output format
    out = [list(all_edges[idx]) for idx in sorted(best_set)]
    return out
# EVOLVE-BLOCK-END
