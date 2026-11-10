import random
import time

def construct():
    n = 40
    adj = [0] * n
    edge_count = 0
    
    def creates_c6(u, v):
        if adj[u] & adj[v]:
            return True
        mask_u = 1 << u
        mask_v = 1 << v
        forbidden = mask_u | mask_v
        au = adj[u]
        while au:
            lsb_a = au & -au
            a = lsb_a.bit_length() - 1
            au ^= lsb_a
            bv = adj[v]
            while bv:
                lsb_b = bv & -bv
                b = lsb_b.bit_length() - 1
                bv ^= lsb_b
                if a == b:
                    continue
                if adj[a] & adj[b] & ~forbidden:
                    return True
        return False

    def get_random_existing_edge():
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x != y and (adj[x] & (1 << y)):
                return x, y

    # Greedy Init
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(edges)
    for u, v in edges:
        if not creates_c6(u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edge_count += 1
            
    best_adj = adj[:]
    best_count = edge_count
    
    start_time = time.time()
    timeout = 140.0
    
    stagnation = 0
    threshold = 100
    
    while time.time() - start_time < timeout:
        # Pick random non-edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or (adj[u] & (1 << v)):
            continue
            
        if not creates_c6(u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edge_count += 1
            if edge_count > best_count:
                best_count = edge_count
                best_adj = adj[:]
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > threshold:
                # Remove 5 random edges to perturb
                for _ in range(5):
                    x, y = get_random_existing_edge()
                    adj[x] ^= 1 << y
                    adj[y] ^= 1 << x
                    edge_count -= 1
                stagnation = 0

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result
