import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 31
    target_edges = 91
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_c4(adj_mat):
        # adj_mat is 0/1 numpy array
        A2 = np.dot(adj_mat, adj_mat)
        # Extract upper triangle
        triu = np.triu(A2, k=1)
        # Calculate C(k, 2) for each entry
        # k = triu[i,j]
        # C(k, 2) = k * (k-1) / 2
        # Using integer arithmetic
        vals = triu * (triu - 1) // 2
        total = np.sum(vals)
        return total // 2

    best_score = float('inf')
    best_edges = []
    
    # Multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        
        # Initialize random graph
        random.shuffle(all_pairs)
        initial_edges = all_pairs[:target_edges]
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in initial_edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        # Build lists for fast swap
        edge_set = set(initial_edges)
        e_list = list(initial_edges)
        ne_list = [e for e in all_pairs if e not in edge_set]
        
        current_score = count_c4(adj)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        
        while time.time() < deadline - 5.0 and T > min_T:
            if not e_list or not ne_list:
                break
                
            # Pick random edge to remove
            e_idx = random.randint(0, len(e_list) - 1)
            u, v = e_list[e_idx]
            
            # Pick random non-edge to add
            ne_idx = random.randint(0, len(ne_list) - 1)
            a, b = ne_list[ne_idx]
            
            # Perform swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1
            
            new_score = count_c4(adj)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_score = new_score
                # Update lists
                e_list[e_idx] = (a, b)
                ne_list[ne_idx] = (u, v)
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [list(e) for e in e_list]
            else:
                # Reject
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
            T *= cooling

    if not best_edges:
        # Fallback: return the polarity graph + best edge if SA failed
        # But SA should always produce something
        pass

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Generated {len(edges)} edges")
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A2 = np.dot(adj, adj)
    np.fill_diagonal(A2, 0)
    prod = A2 * (A2 - 1)
    sum_upper = np.sum(np.triu(prod, 1))
    c4s = sum_upper // 4
    print(f"C4 count: {c4s}")
