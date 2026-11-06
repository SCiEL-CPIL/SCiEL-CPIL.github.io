import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            p1 = a - 1
            p2 = a + d - 1
            p3 = a + 2*d - 1
            p4 = a + 3*d - 1
            
            ap_id = len(aps)
            aps.append((p1, p2, p3, p4))
            
            index_to_aps[p1].append(ap_id)
            index_to_aps[p2].append(ap_id)
            index_to_aps[p3].append(ap_id)
            index_to_aps[p4].append(ap_id)
            
    # We want to minimize K. Try K = 3, 4, ...
    # K=1,2 are impossible.
    
    def try_k(K):
        colors = [-1] * N
        nodes_visited = 0
        max_nodes = 500000  # Limit search size
        
        def backtrack(idx):
            nonlocal nodes_visited
            if time.time() > deadline:
                return False
            if nodes_visited > max_nodes:
                return False
            nodes_visited += 1
            
            if idx == N:
                return True
            
            # Try each color 0..K-1
            # Heuristic: Try colors that appear in the APs most recently?
            # Or just 0..K-1.
            for c in range(K):
                valid = True
                for ap_id in index_to_aps[idx]:
                    p1, p2, p3, p4 = aps[ap_id]
                    # Only check if idx is the last (largest) index in the AP
                    if idx != p4:
                        continue
                    
                    c1 = colors[p1]
                    c2 = colors[p2]
                    c3 = colors[p3]
                    
                    # Check if {c1, c2, c3, c} has at least 3 distinct colors
                    # Using a set is fast enough for small K
                    if len({c1, c2, c3, c}) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            
            return False
        
        return backtrack(0), colors

    for K in range(3, 128):
        success, colors = try_k(K)
        if success:
            return colors
        
    # Fallback: Greedy with 4 colors
    colors = [-1] * N
    for i in range(N):
        for c in range(4):
            valid = True
            for ap_id in index_to_aps[i]:
                p1, p2, p3, p4 = aps[ap_id]
                if i != p4:
                    continue
                c1 = colors[p1]
                c2 = colors[p2]
                c3 = colors[p3]
                if len({c1, c2, c3, c}) < 3:
                    valid = False
                    break
            if valid:
                colors[i] = c
                break
        else:
            colors[i] = 0 # Should not happen
            
    return colors
