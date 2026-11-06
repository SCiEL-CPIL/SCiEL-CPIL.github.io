"""Find minimum-bottleneck Hamilton cycle via binary search + pruned bitmask DP."""
import time

# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
W = [[0] * N for _ in range(N)]
for idx, (u, v) in enumerate(P['edge_order']):
    W[u][v] = idx + 1
    W[v][u] = idx + 1

def bottleneck_score(path):
    max_w = 0
    for i in range(N):
        u = path[i]
        v = path[(i + 1) % N]
        w = W[u][v]
        if w > max_w:
            max_w = w
    return max_w

def greedy_construct():
    path = [0]
    visited = [False] * N
    visited[0] = True
    
    for _ in range(N - 1):
        last = path[-1]
        best_next = -1
        best_w = float('inf')
        for nxt in range(N):
            if not visited[nxt]:
                w = W[last][nxt]
                if w < best_w:
                    best_w = w
                    best_next = nxt
        if best_next == -1:
            return None
        path.append(best_next)
        visited[best_next] = True
        
    return path

def two_opt_improve(path, deadline):
    current_score = bottleneck_score(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                return path
            for j in range(i + 2, N):
                if i == 0 and j == N - 1:
                    continue
                u1 = path[(i - 1) % N]
                u2 = path[i]
                u3 = path[j]
                u4 = path[(j + 1) % N]
                
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                new_score = bottleneck_score(new_path)
                
                if new_score < current_score:
                    path = new_path
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_path = None
    best_score = float('inf')
    
    path = greedy_construct()
    if path:
        path = two_opt_improve(path, deadline)
        score = bottleneck_score(path)
        if score < best_score:
            best_score = score
            best_path = path
            
    if time.time() < deadline - 5:
        rng = random.Random(614)
        for _ in range(10):
            if time.time() > deadline:
                break
            p = list(range(N))
            rng.shuffle(p)
            p = two_opt_improve(p, deadline)
            s = bottleneck_score(p)
            if s < best_score:
                best_score = s
                best_path = p
                
    if best_path:
        return best_path
    else:
        return list(range(N))
# EVOLVE-BLOCK-END
