"""K3,3-free graph construction on 20 vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 150s kill

    n = 20

    def count_common_neighbors(adj, i, j, k):
        """Count common neighbors of vertices i, j, k (excluding themselves)."""
        common = adj[i] & adj[j] & adj[k]
        # Remove i, j, k from consideration (they can't be their own neighbors in simple graph,
        # but let's be safe)
        common &= ~(1 << i)
        common &= ~(1 << j)
        common &= ~(1 << k)
        return common.bit_count()

    def is_k33_free_fast(adj):
        """Check if graph is K3,3-free using precomputed pair common neighbors."""
        # Precompute common neighbors for all pairs
        pair_common = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i] & adj[j]
                c &= ~(1 << i)
                c &= ~(1 << j)
                pair_common[i][j] = c
                pair_common[j][i] = c
        
        # For each triple (i,j,k), check if common neighbors >= 3
        for i in range(n):
            for j in range(i+1, n):
                c_ij = pair_common[i][j]
                for k in range(j+1, n):
                    c_ijk = c_ij & adj[k]
                    c_ijk &= ~(1 << k)
                    if c_ijk.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v), return True if graph remains K3,3-free."""
        old_u = adj[u]
        old_v = adj[v]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if not is_k33_free_fast(adj):
            adj[u] = old_u
            adj[v] = old_v
            return False
        return True

    def try_swap(adj, remove_pair, add_pair):
        """Try to remove edge remove_pair and add add_pair."""
        u1, v1 = remove_pair
        u2, v2 = add_pair
        old_u1 = adj[u1]
        old_v1 = adj[v1]
        old_u2 = adj[u2]
        old_v2 = adj[v2]
        
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        
        if not is_k33_free_fast(adj):
            adj[u1] = old_u1
            adj[v1] = old_v1
            adj[u2] = old_u2
            adj[v2] = old_v2
            return False
        return True

    def paley_graph_19():
        """Construct Paley graph on 19 vertices (quadratic residues mod 19)."""
        p = 19
        adj = [0] * p
        for a in range(p):
            for b in range(a+1, p):
                diff = (b - a) % p
                if diff != 0 and (diff * diff) % p == 1:  # quadratic residue
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        return adj

    def embed_19_to_20(adj19):
        """Embed 19-vertex graph into 20 vertices by adding vertex 19 connected to some vertices."""
        adj20 = [0] * 20
        for i in range(19):
            adj20[i] = adj19[i]
        
        # Vertex 19: try connecting to vertices that maintain K3,3-free property
        # Start with no connections, then greedily add
        adj20[19] = 0
        # Try adding edges from 19 to each vertex
        for v in range(19):
            if try_add_edge(adj20, 19, v):
                pass  # edge added
        
        return adj20

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    # Strategy 1: Start from Paley graph on 19 vertices, embed to 20
    random.seed(42)
    
    adj19 = paley_graph_19()
    adj20 = embed_19_to_20(adj19)
    current_count = count_edges(adj20)
    if current_count > best_count:
        best_count = current_count
        best_adj = adj20[:]
        best_edges = adj_to_edges(adj20)
        print(f"Paley+embed: {best_count} edges", flush=True)

    # Strategy 2: Local search from best so far
    adj = best_adj[:]
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Multiple rounds of local search
    for round_num in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Build lists of edges and non-edges
        edges_list = []
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    edges_list.append((u, v))
                else:
                    non_edges.append((u, v))
        
        # Try random swaps
        random.shuffle(non_edges)
        random.shuffle(edges_list)
        
        # Limit attempts per round to keep time bounded
        max_attempts = min(len(non_edges) * 2, 200)
        attempts = 0
        
        for ne in non_edges:
            if attempts >= max_attempts or time.time() > deadline - 5:
                break
            for e in edges_list:
                if attempts >= max_attempts or time.time() > deadline - 5:
                    break
                attempts += 1
                if ne[0] == e[0] and ne[1] == e[1]:
                    continue
                if try_swap(adj, e, ne):
                    improved = True
                    break
            if improved:
                break
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)
            print(f"Round {round_num}: {best_count} edges", flush=True)
        
        if not improved and round_num > 10:
            break

    # Fallback: if no edges found, return K2,18
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges
# EVOLVE-BLOCK-END
