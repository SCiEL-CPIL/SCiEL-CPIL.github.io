import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 8-vertex subsets as bitmasks
    subsets_8 = list(itertools.combinations(vertices, 8))
    
    # Precompute for each 8-subset: list of 70 partitions (as two 4-vertex tuples)
    # Partition index: which 4 of the 8 positions go to X
    partition_indices = list(itertools.combinations(range(8), 4))
    
    # For fast Q3 check: use bitmask adjacency
    # adj_mask[v] is a 16-bit integer where bit u is set if edge (v,u) exists
    
    def has_q3_fast(adj_mask):
        """Check if graph (represented as list of 16-bit masks) contains Q3."""
        for s in subsets_8:
            # Count edges in s using bitmasks
            edge_count = 0
            for i in range(8):
                v = s[i]
                # Count neighbors in s that come after i
                mask = adj_mask[v]
                for j in range(i+1, 8):
                    if mask & (1 << s[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            if edge_count > 12:
                # Q3 has exactly 12 edges; more than 12 means it's not a subgraph
                # Wait - we need to check if Q3 is a SUBGRAPH, not induced subgraph
                # So we need at least 12 edges, and we need to find a subset of 12 edges forming Q3
                # This is harder. Let's just check if there exists a partition where
                # each vertex has degree >= 3 in the bipartite sense
                pass
            
            # For each partition into X, Y of size 4
            # Check if the bipartite graph between X and Y has min degree >= 3
            # This means Q3 is a subgraph (not necessarily induced)
            for x_combo in partition_indices:
                x_set = set(x_combo)
                y_set = set(range(8)) - x_set
                
                # Check min degree from X side
                valid = True
                for i in x_set:
                    v = s[i]
                    deg = 0
                    for j in y_set:
                        if adj_mask[v] & (1 << s[j]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check min degree from Y side
                for j in y_set:
                    u = s[j]
                    deg = 0
                    for i in x_set:
                        if adj_mask[u] & (1 << s[i]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False
    
    def build_adj_mask(edge_set):
        """Build adjacency mask list from edge set."""
        adj_mask = [0] * n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Try Shrikhande graph complement as initial candidate
    # Shrikhande graph: strongly regular (16, 6, 2, 2)
    # Complement has degree 16-1-6=9, so 16*9/2=72 edges
    # Actually let me construct the Shrikhande graph properly
    
    # Shrikhande graph vertices: Z_4 x Z_4
    # Two vertices (a,b) and (c,d) are adjacent if:
    # - They differ in exactly one coordinate by 1 mod 4, OR
    # - They differ in both coordinates by 2 mod 4 (i.e., (a+2,b+2) or (a-2,b-2))
    
    def shrikhande_adj(u, v):
        """Check if u,v are adjacent in Shrikhande graph. u,v in 0..15."""
        a, b = divmod(u, 4)
        c, d = divmod(v, 4)
        da = (a - c) % 4
        db = (b - d) % 4
        # Adjacent if:
        # 1. Same row, adjacent col: da==0 and (db==1 or db==3)
        # 2. Same col, adjacent row: db==0 and (da==1 or da==3)
        # 3. Diagonal: da==2 and db==2
        if da == 0 and (db == 1 or db == 3):
            return True
        if db == 0 and (da == 1 or da == 3):
            return True
        if da == 2 and db == 2:
            return True
        return False
    
    # Build Shrikhande graph edges
    shrikhande_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            if shrikhande_adj(u, v):
                shrikhande_edges.add((u, v))
    
    # Complement of Shrikhande
    all_edges = set((u, v) for u in range(16) for v in range(u+1, 16))
    complement_edges = all_edges - shrikhande_edges
    
    # Check if complement is Q3-free
    adj_mask = build_adj_mask(complement_edges)
    if not has_q3_fast(adj_mask):
        best_edges = set(complement_edges)
        best_count = len(best_edges)
    else:
        # Fall back to K_3,13 + triangle
        best_edges = set()
        for a in range(3):
            for b in range(3, 16):
                best_edges.add((a, b))
        for u, v in [(0,1), (1,2), (0,2)]:
            best_edges.add((u, v))
        best_count = len(best_edges)
    
    # Local search: try to add edges to improve
    adj_mask = build_adj_mask(best_edges)
    current_edges = set(best_edges)
    
    # Get all possible edges not in current graph
    candidate_edges = list(all_edges - current_edges)
    
    # Multiple passes of greedy addition
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        random.shuffle(candidate_edges)
        
        # Reset to best
        current_edges = set(best_edges)
        adj_mask = build_adj_mask(current_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            
            # Try adding edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            current_edges.add((u, v))
            
            if has_q3_fast(adj_mask):
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                current_edges.discard((u, v))
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
