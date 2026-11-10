import numpy as np
import random

def construct():
    n = 31
    m = 91
    
    def get_edges_from_matrix(A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i][j]:
                    edges.append([i, j])
        return edges

    def count_4_cycles(A):
        A2 = A @ A
        tr_A4 = np.sum(A2 * A2)
        degrees = np.sum(A, axis=1)
        sum_d2 = np.sum(degrees * degrees)
        m_val = np.sum(degrees) // 2
        return int((tr_A4 - 2 * sum_d2 + 2 * m_val) / 8)

    best_score = float('inf')
    best_edges = None
    
    num_restarts = 50
    steps_per_restart = 1000
    
    for r in range(num_restarts):
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible_edges)
        initial_edges = all_possible_edges[:m]
        
        A = np.zeros((n, n), dtype=int)
        for u, v in initial_edges:
            A[u][v] = 1
            A[v][u] = 1
            
        current_score = count_4_cycles(A)
        
        for step in range(steps_per_restart):
            idx_remove = random.randint(0, len(initial_edges) - 1)
            u, v = initial_edges[idx_remove]
            
            A[u][v] = 0
            A[v][u] = 0
            initial_edges.pop(idx_remove)
            
            found = False
            while not found:
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j and A[i][j] == 0:
                    if (i == u and j == v) or (i == v and j == u):
                        continue
                    found = True
            
            A[i][j] = 1
            A[j][i] = 1
            initial_edges.append([i, j])
            
            new_score = count_4_cycles(A)
            
            if new_score < current_score:
                current_score = new_score
            else:
                A[i][j] = 0
                A[j][i] = 0
                initial_edges.pop()
                A[u][v] = 1
                A[v][u] = 1
                initial_edges.insert(idx_remove, [u, v])
                
        if current_score < best_score:
            best_score = current_score
            best_edges = get_edges_from_matrix(A)
            
    return best_edges
