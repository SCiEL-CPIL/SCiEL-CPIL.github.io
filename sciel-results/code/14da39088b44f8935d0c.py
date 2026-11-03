import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3 and g % 2 == 0 and g >= 4:
        # Moore bound
        lower_bound = 2 * (2 ** (g // 2 - 1) - 1)
        if lower_bound % 2 != 0:
            lower_bound += 1
            
        # Search Circulants
        # n even
        # Steps {n//2, a, -a}
        
        def check_girth_circulant(n, a):
            # BFS from 0
            visited = [-1] * n
            visited[0] = 0
            queue = [0]
            max_depth = g // 2 - 1
            s1 = n // 2
            for depth in range(max_depth + 1):
                next_queue = []
                for u in queue:
                    for s in [s1, a, -a]:
                        v = (u + s) % n
                        if visited[v] == -1:
                            visited[v] = depth + 1
                            next_queue.append(v)
                        elif visited[v] < depth:
                            if depth + 2 + visited[v] < g:
                                return False
                queue = next_queue
                if not queue:
                    break
            return True

        for n in range(lower_bound, 1500, 2):
            if time.time() > deadline - 20:
                break
            for _ in range(1000):
                a = random.randint(1, n // 2 - 1)
                if check_girth_circulant(n, a):
                    edges = []
                    for i in range(n):
                        for j in [(i + n//2) % n, (i + a) % n, (i - a) % n]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

        # Search Generalized Petersen
        # n_half such that 2*n_half is total vertices
        # Total vertices N = 2 * n_half
        # Search n_half from lower_bound//2 to 750
        
        def check_girth_gp(n_half, k_gp):
            N = 2 * n_half
            visited = [-1] * N
            # Map (i, 0) to i, (i, 1) to i + n_half
            visited[0] = 0
            queue = [0]
            max_depth = g // 2 - 1
            
            def get_neighbors(v):
                i = v % n_half
                side = v // n_half
                if side == 0:
                    return [(i + 1) % n_half, i, i + n_half + k_gp % n_half] # Wait, (i,1) is i+n_half. Neighbor (i+k,1) is i+k+n_half.
                    # Correction:
                    # (i,0) neighbors: (i+1,0), (i-1,0), (i,1)
                    # (i,1) neighbors: (i,0), (i+k,1), (i-k,1)
                    # Let's just generate them.
                    pass
            
            # Simplified BFS for GP
            # Node representation: (i, side)
            # Index: i if side 0, i+n_half if side 1
            
            visited = [-1] * N
            visited[0] = 0 # (0,0)
            queue = [0]
            
            for depth in range(max_depth + 1):
                next_queue = []
                for u in queue:
                    i = u % n_half
                    side = u // n_half
                    if side == 0:
                        # Neighbors: (i+1,0), (i-1,0), (i,1)
                        neighbors = [
                            (i + 1) % n_half,
                            (i - 1) % n_half,
                            i + n_half
                        ]
                    else:
                        # Neighbors: (i,0), (i+k,1), (i-k,1)
                        neighbors = [
                            i,
                            ((i + k_gp) % n_half) + n_half,
                            ((i - k_gp) % n_half) + n_half
                        ]
                    
                    for v in neighbors:
                        if visited[v] == -1:
                            visited[v] = depth + 1
                            next_queue.append(v)
                        elif visited[v] < depth:
                            if depth + 2 + visited[v] < g:
                                return False
                queue = next_queue
                if not queue:
                    break
            return True

        for n_half in range(lower_bound // 2, 750):
            if time.time() > deadline - 5:
                break
            for _ in range(500):
                k_gp = random.randint(1, n_half - 1)
                if check_girth_gp(n_half, k_gp):
                    N = 2 * n_half
                    edges = []
                    for i in range(n_half):
                        # Outer cycle
                        edges.append((i, (i + 1) % n_half))
                        # Spokes
                        edges.append((i, i + n_half))
                        # Inner star
                        edges.append((i + n_half, ((i + k_gp) % n_half) + n_half))
                    return N, edges

    # Fallback LUW
    # ... (copy existing LUW code) ...
