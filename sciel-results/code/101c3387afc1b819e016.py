import time
import random
import itertools
import sys

# Q3 structure
Q3_ADJ = [
    [1, 2, 4],
    [0, 3, 5],
    [0, 3, 6],
    [1, 2, 7],
    [0, 5, 6],
    [1, 4, 7],
    [2, 4, 7],
    [3, 5, 6]
]

Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

# Precompute for each Q3 vertex, the set of Q3 neighbors as a bitmask
Q3_NEIGHBOR_MASK = []
for i in range(8):
    mask = 0
    for nb in Q3_ADJ[i]:
        mask |= (1 << nb)
    Q3_NEIGHBOR_MASK.append(mask)

# For each Q3 edge (i,j), precompute the mask of other Q3 vertices that are neighbors of i or j
# This helps in pruning

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 150s limit
    
    n = 16
    all_vertices = list(range(n))
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Fast Q3 check using bitmasks
    # adj_mask[i] is a 16-bit integer where bit j is set if (i,j) is an edge
    
    def creates_q3_fast(u, v, adj_mask):
        """Check if adding edge (u,v) creates a Q3 subgraph.
        Uses bitmask-based backtracking for speed."""
        
        # For each Q3 edge (qi, qj), try mapping u->qi, v->qj and u->qj, v->qi
        # Then backtrack to assign remaining 6 Q3 vertices
        
        # We'll use iterative deepening with bitmasks
        # assigned_mask: 8-bit mask of which Q3 vertices are assigned
        # assign_to: list of length 8, assign_to[q] = g_vertex or -1
        
        def backtrack(assigned_mask, assign_to, available_mask):
            # Find unassigned Q3 vertex with most assigned neighbors
            best_q = -1
            best_constraint = -1
            for q in range(8):
                if not (assigned_mask & (1 << q)):
                    # Count assigned neighbors of q in Q3
                    nb_mask = Q3_NEIGHBOR_MASK[q]
                    assigned_nbs = nb_mask & assigned_mask
                    count = bin(assigned_nbs).count('1')
                    if count > best_constraint:
                        best_constraint = count
                        best_q = q
            
            if best_q == -1:
                return True  # All assigned
            
            # Get candidates from available vertices that are adjacent to all assigned neighbors
            nb_mask = Q3_NEIGHBOR_MASK[best_q]
            assigned_nbs = nb_mask & assigned_mask
            
            # Start with all available vertices
            candidates = available_mask
            
            # Filter: must be adjacent to each assigned neighbor
            assigned_nb_list = []
            temp = assigned_nbs
            while temp:
                lsb = temp & (-temp)
                qnb = lsb.bit_length() - 1
                assigned_nb_list.append(assign_to[qnb])
                temp ^= lsb
            
            for gnb in assigned_nb_list:
                # candidates must have bit gnb set in their adjacency
                candidates &= adj_mask[gnb]
                if candidates == 0:
                    return False
            
            # Try each candidate
            while candidates:
                lsb = candidates & (-candidates)
                g_cand = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_assigned_mask = assigned_mask | (1 << best_q)
                new_assign_to = assign_to[:]
                new_assign_to[best_q] = g_cand
                new_available_mask = available_mask & ~(1 << g_cand)
                
                if backtrack(new_assigned_mask, new_assign_to, new_available_mask):
                    return True
            
            return False
        
        # Try all Q3 edges as the one corresponding to (u,v)
        for qi, qj in Q3_EDGES:
            # Case 1: u->qi, v->qj
            # Check that u and v would be adjacent (we're adding this edge, so assume yes)
            # But we need to verify that the initial assignment is consistent with existing edges
            # The edge (qi,qj) in Q3 corresponds to (u,v) which we're adding
            # Other edges from qi to other Q3 vertices must have u adjacent to their eventual images
            # Other edges from qj to other Q3 vertices must have v adjacent to their eventual images
            # These are handled by the backtracking
            
            assign_to = [-1] * 8
            assign_to[qi] = u
            assign_to[qj] = v
            assigned_mask = (1 << qi) | (1 << qj)
            available_mask = ((1 << n) - 1) & ~(1 << u) & ~(1 << v)
            
            if backtrack(assigned_mask, assign_to, available_mask):
                return True
            
            # Case 2: u->qj, v->qi
            assign_to = [-1] * 8
            assign_to[qj] = u
            assign_to[qi] = v
            assigned_mask = (1 << qi) | (1 << qj)
            available_mask = ((1 << n) - 1) & ~(1 << u) & ~(1 << v)
            
            if backtrack(assigned_mask, assign_to, available_mask):
                return True
        
        return False
    
    def count_edges(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2
    
    def random_greedy_init(seed):
        random.seed(seed)
        adj_mask = [0] * n
        edges = list(all_edges)
        random.shuffle(edges)
        
        for u, v in edges:
            if not creates_q3_fast(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        return adj_mask
    
    best_adj = None
    best_count = 0
    
    # Run multiple restarts with time budget
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        # 1. Initialize with random greedy
        adj = random_greedy_init(restart_count * 7919)
        current_count = count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # 2. Local search: try to add edges
        # Build list of missing edges once, then shuffle and try
        missing_edges = []
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                missing_edges.append((u, v))
        
        random.shuffle(missing_edges)
        
        # Try to add as many edges as possible
        for u, v in missing_edges:
            if time.time() > deadline - 5:
                break
            if not ((adj[u] >> v) & 1):  # Still missing?
                if not creates_q3_fast(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # 3. If we have time, do a more aggressive local search
        if time.time() < deadline - 10:
            # Try swapping: remove one edge, add two
            existing_edges = []
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    existing_edges.append((u, v))
            
            random.shuffle(existing_edges)
            
            for ru, rv in existing_edges[:5]:
                if time.time() > deadline - 5:
                    break
                
                # Remove edge (ru, rv)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                current_count -= 1
                
                # Try to add up to 2 new edges
                missing = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
                random.shuffle(missing)
                
                added = 0
                for u, v in missing:
                    if added >= 2:
                        break
                    if not creates_q3_fast(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_count += 1
                        added += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                
                # If we didn't gain, put the edge back
                if added < 2:
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    current_count += 1
    
    # Convert best_adj to list of edges
    if best_adj is None:
        return []
    
    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
    
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
