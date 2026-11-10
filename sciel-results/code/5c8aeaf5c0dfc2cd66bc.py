import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n = 20
    np.random.seed(42)
    
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    triples = np.array(triples)
    t_count = len(triples)
    
    edge_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    e_count = len(edge_pairs)
    
    def compute_penalty(A):
        cn = (A[triples[:,0]] & A[triples[:,1]] & A[triples[:,2]]).sum(axis=1)
        return int(np.sum(np.maximum(cn - 2, 0)))
    
    def greedy_complete(A):
        """Try to add every missing edge if it doesn't increase penalty."""
        improved = True
        while improved:
            improved = False
            for i, j in edge_pairs:
                if A[i, j] == 0:
                    A[i, j] = 1
                    A[j, i] = 1
                    new_pen = compute_penalty(A)
                    if new_pen > 0:
                        A[i, j] = 0
                        A[j, i] = 0
                    else:
                        improved = True
        return A
    
    best_edges = []
    best_edge_count = 0
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random start
        A = np.zeros((n, n), dtype=np.int8)
        p = 0.5 + 0.1 * restart
        for i in range(n):
            for j in range(i+1, n):
                if np.random.random() < p:
                    A[i, j] = 1
                    A[j, i] = 1
        
        current_penalty = compute_penalty(A)
        current_edges = int(np.triu(A).sum())
        current_obj = current_edges - 1000 * current_penalty
        
        best_A = A.copy()
        best_obj = current_obj
        best_edges_in_sa = current_edges
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        while T > T_min and time.time() < deadline - 30:
            idx = np.random.randint(0, e_count)
            i, j = edge_pairs[idx]
            
            old_val = A[i, j]
            new_val = 1 - old_val
            
            A[i, j] = new_val
            A[j, i] = new_val
            
            new_penalty = compute_penalty(A)
            new_edges = int(np.triu(A).sum())
            new_obj = new_edges - 1000 * new_penalty
            
            delta = new_obj - current_obj
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_obj = new_obj
                current_penalty = new_penalty
                current_edges = new_edges
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_edges_in_sa = current_edges
                    best_A = A.copy()
            else:
                A[i, j] = old_val
                A[j, i] = old_val
            
            T *= cooling
        
        # Greedy completion
        best_A = greedy_complete(best_A)
        final_edges = int(np.triu(best_A).sum())
        
        if final_edges > best_edge_count:
            best_edge_count = final_edges
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if best_A[i, j]]
    
    return best_edges
