import time
import sys

# Increase recursion limit for Bron-Kerbosch
sys.setrecursionlimit(10000)

DIMENSIONS = [3, 5, 7]

def _bron_kerbosch(R, P, X, adj, best_set):
    """
    Bron-Kerbosch algorithm with pivoting to find the maximum independent set.
    R: set of vertices in the current independent set
    P: set of candidates that are adjacent to all vertices in R (in the complement graph, 
       this means they are non-conflicting with R in the original conflict graph context? 
       Wait. We want an Independent Set in the Conflict Graph.
       So we are looking for a clique in the Complement Graph.
       Or we can just run BK on the Conflict Graph looking for Independent Set? 
       Standard BK finds Cliques. 
       If we want Max Independent Set in G, we find Max Clique in G_complement.
       
       Let's stick to finding Max Independent Set in the Conflict Graph directly.
       R: current independent set.
       P: vertices that can be added to R (non-conflicting with all in R).
       X: vertices that have already been processed (to avoid duplicates).
       
       Condition for P: v in P must not conflict with any u in R.
    """
    if not P and not X:
        if len(R) > len(best_set):
            best_set.clear()
            best_set.update(R)
        return

    # Pivoting: choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
    # In the context of Independent Set, N(u) are neighbors in the Conflict Graph.
    # We want to select vertices from P \ N(u) to branch on.
    # Why? Because any independent set containing u must not contain its neighbors.
    # So if we branch on P \ N(u), we cover all possibilities.
    
    P_union_X = P.union(X)
    if not P_union_X:
        return
        
    # Find pivot u
    pivot = None
    max_neighbors = -1
    for u in P_union_X:
        # Count neighbors of u in P
        # adj[u] is the set of vertices conflicting with u
        num_neighbors = len(P.intersection(adj[u]))
        if num_neighbors > max_neighbors:
            max_neighbors = num_neighbors
            pivot = u
            
    # Branch on P \ N(pivot)
    candidates = P.difference(adj[pivot])
    
    for v in list(candidates):
        # New R
        new_R = R.union({v})
        # New P: vertices in P that do not conflict with v
        new_P = P.intersection(set(adj[v].symmetric_difference(set(adj[v]))) if False else P) # P minus neighbors of v
        # Actually: new_P = { w in P | w does not conflict with v }
        # Since adj[v] contains conflicts, new_P = P - adj[v]
        new_P = P.difference(adj[v])
        # New X: vertices in X that do not conflict with v
        new_X = X.difference(adj[v])
        
        _bron_kerbosch(new_R, new_P, new_X, adj, best_set)
        
        # Move v from P to X
        P.remove(v)
        X.add(v)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # 1. Generate all possible 2x2x2 Strassen blocks
    # A block is defined by (i1, i2, j1, j2, k1, k2)
    all_blocks = []
    block_indices_list = []
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices_list.append(indices)

    num_blocks = len(all_blocks)
    
    # 2. Build Conflict Graph
    # adj[i] is a set of indices j such that block i and block j conflict (share an index)
    adj = [set() for _ in range(num_blocks)]
    
    # Optimization: Instead of O(N^2) set intersection, we can map indices to blocks
    # But N=630, so O(N^2) is ~400k operations, which is fast in Python if optimized.
    # Let's use a mapping from index to list of blocks containing it.
    index_to_blocks = {}
    for b_idx, indices in enumerate(block_indices_list):
        for idx in indices:
            if idx not in index_to_blocks:
                index_to_blocks[idx] = []
            index_to_blocks[idx].append(b_idx)

    for idx, blocks_containing_idx in index_to_blocks.items():
        # All pairs in this list conflict
        for i in range(len(blocks_containing_idx)):
            for j in range(i + 1, len(blocks_containing_idx)):
                u = blocks_containing_idx[i]
                v = blocks_containing_idx[j]
                adj[u].add(v)
                adj[v].add(u)

    # 3. Find Maximum Independent Set (MIS)
    # We want a set of blocks that do NOT conflict.
    # This is the Max Independent Set in the conflict graph.
    # This is equivalent to Max Clique in the complement graph.
    # The Bron-Kerbosch implementation above finds Max Independent Set directly 
    # by branching on non-neighbors.
    
    best_set = set()
    # Initial P: all blocks
    P = set(range(num_blocks))
    X = set()
    R = set()
    
    # Run BK
    _bron_kerbosch(R, P, X, adj, best_set)
    
    num_blocks_used = len(best_set)
    
    # 4. Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for b_idx in best_set:
        i1, i2, j1, j2, k1, k2 = all_blocks[b_idx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered.update(block_indices_list[b_idx])

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
