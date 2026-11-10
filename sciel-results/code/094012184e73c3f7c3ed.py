import time
import random
import math
from itertools import combinations, product

# Precompute all 8-vertex subsets and their Q3-free edge capacities
# We represent a graph on 8 vertices by a 28-bit mask (edges (0,1)..(6,7))
# We precompute for every 28-bit mask whether it is Q3-free.
# There are 2^28 = 268,435,456 masks. This is too many to precompute fully in Python quickly.
# Instead, we precompute a set of all Q3 subgraphs on 8 vertices (canonical labels 0..7).
# A mask is Q3-free if it does not contain any of these.

# Q3 edges in canonical order:
# (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
# Map these to indices in the 28-edge list for vertices 0..7.
# Edge index for (u,v) with u<v is u*7 - u*(u-1)//2 + (v-u-1)
def edge_idx_8(u, v):
    if u > v:
        u, v = v, u
    # Number of edges starting with 0: 7
    # Number of edges starting with 1: 6
    # ...
    # Index = sum_{k=0}^{u-1} (7-k) + (v-u-1)
    # = 7u - u(u-1)/2 + v - u - 1
    # = 6u - u(u-1)/2 + v - 1
    return 6*u - u*(u-1)//2 + v - 1

Q3_CANON_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

# Generate all distinct Q3 subgraphs on vertices 0..7
# We generate all 8! permutations of vertices 0..7, map the canonical Q3 edges to the permuted vertices,
# and store the resulting 28-bit mask.
q3_masks = set()
from itertools import permutations
for perm in permutations(range(8)):
    mask = 0
    valid = True
    for u, v in Q3_CANON_EDGES:
        pu, pv = perm[u], perm[v]
        if pu > pv:
            pu, pv = pv, pu
        idx = edge_idx_8(pu, pv)
        mask |= (1 << idx)
    q3_masks.add(mask)

# q3_masks contains all edge sets that form a Q3 on some labeling of 8 vertices.
# Note: This set has size 8! / 48 = 1344? No, it's the number of distinct edge sets.
# The automorphism group of Q3 has size 48. So there are 1344 distinct embeddings?
# Actually, the number of distinct Q3 subgraphs in K8 is C(8,8) * (8!/48) = 1344.
# So we have a set of 1344 masks.

# To check if a mask is Q3-free:
# A mask M is Q3-free if for all m in q3_masks, (M & m) != m.
# This is 1344 checks per mask.

# We will use this to check if a specific 8-vertex induced subgraph is Q3-free.
# For the partition strategy, we need to know if we can add an edge within a partition.
# If a partition has size k, we can check all 8-subsets of it.

def is_q3_free_8(mask):
    """Check if a 28-bit mask on vertices 0..7 is Q3-free."""
    for qm in q3_masks:
        if (mask & qm) == qm:
            return False
    return True

def construct():
    deadline = time.time() + 115
    n = 16
    full_mask = (1 << n) - 1
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
    
    # Function to check if adding edge (u,v) to a graph creates a Q3
    # We do this by checking all 8-vertex subsets containing u and v.
    # If the induced subgraph on those 8 vertices contains a Q3, and the Q3 uses the edge (u,v),
    # then we have a problem. But it's easier to just check if the new graph has ANY Q3.
    # However, we only care if the NEW edge creates a Q3.
    # So we check all 8-subsets S containing u and v.
    # For each S, check if the induced subgraph G[S] is Q3-free.
    # If any G[S] is not Q3-free, we reject.
    
    def creates_q3(u, v, adj):
        # Find all vertices w such that w != u, v
        others = [w for w in range(n) if w != u and w != v]
        # We need to choose 6 more vertices from 'others' to form an 8-set.
        # C(14, 6) = 3003. This is fast enough.
        for combo in combinations(others, 6):
            S = sorted([u, v] + list(combo))
            # Build mask for induced subgraph on S
            # Map S[0]..S[7] to 0..7
            mapping = {S[i]: i for i in range(8)}
            mask = 0
            for i in range(8):
                for j in range(i+1, 8):
                    ui, vi = S[i], S[j]
                    if adj[ui] & (1 << vi):
                        mask |= (1 << edge_idx_8(i, j))
            if not is_q3_free_8(mask):
                return True
        return False
    
    def count_edges(adj):
        count = 0
        for u in range(n):
            count += adj[u].bit_count()
        return count // 2
        
    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    # Strategy:
    # 1. Try random bipartitions (A, B).
    # 2. Start with complete bipartite graph between A and B.
    # 3. Add as many edges as possible within A and within B.
    # 4. Use local search to improve the partition and edge set.
    
    best_adj = [0] * n
    best_count = -1
    
    # Helper to build graph from partition
    def build_from_partition(A, B):
        adj = [0] * n
        # Add all inter-partition edges
        for u in A:
            for v in B:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Add intra-partition edges greedily
        # We want to maximize edges.
        # We can try all possible intra-partition edges in random order.
        intra_edges_A = [(u, v) for u in A for v in A if u < v]
        intra_edges_B = [(u, v) for u in B for v in B if u < v]
        all_intra = intra_edges_A + intra_edges_B
        random.shuffle(all_intra)
        
        for u, v in all_intra:
            if not (adj[u] & (1 << v)):
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj, count_edges(adj)
    
    # Local search on partitions
    def improve_partition(A, B, adj, count, steps=2000):
        best_adj = adj[:]
        best_count = count
        cur_adj = adj[:]
        cur_count = count
        
        for _ in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Move a random vertex from A to B or B to A
            if random.random() < 0.5:
                if not A: continue
                v = random.choice(A)
                A.remove(v)
                B.append(v)
            else:
                if not B: continue
                v = random.choice(B)
                B.remove(v)
                A.append(v)
            
            # Rebuild graph? That's expensive (O(N^2) + Q3 checks).
            # Instead, let's just do a full rebuild for simplicity in this step, 
            # but limit steps.
            # Actually, rebuilding is O(|A||B| + |A|^2 + |B|^2) + Q3 checks.
            # Q3 checks are the bottleneck.
            # Let's just do a few random moves and rebuild.
            
            new_adj, new_count = build_from_partition(A, B)
            
            # Accept if better or with some probability
            if new_count > best_count:
                best_count = new_count
                best_adj = new_adj[:]
                cur_adj = new_adj[:]
                cur_count = new_count
            elif random.random() < 0.1: # Accept worse moves to escape local optima
                cur_adj = new_adj[:]
                cur_count = new_count
            else:
                # Revert move
                if v in B:
                    B.remove(v)
                    A.append(v)
                else:
                    A.remove(v)
                    B.append(v)
        
        return best_adj, best_count

    # Run multiple random starts
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        
        # Random partition
        A = []
        B = []
        for i in range(n):
            if random.random() < 0.5:
                A.append(i)
            else:
                B.append(i)
        
        # Ensure non-empty
        if not A:
            A.append(0)
            if 0 in B: B.remove(0)
        if not B:
            B.append(1)
            if 1 in A: A.remove(1)
        
        adj, cnt = build_from_partition(A, B)
        
        # Local search on partition
        adj, cnt = improve_partition(A, B, adj, cnt, steps=500)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    # Final polish: try adding any remaining edges
    if time.time() < deadline - 2:
        remaining = [(u, v) for u, v in all_edges if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        for u, v in remaining:
            if time.time() > deadline - 1:
                break
            if not (best_adj[u] & (1 << v)) and not creates_q3(u, v, best_adj):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                best_count += 1

    return edges_from_adj(best_adj)
