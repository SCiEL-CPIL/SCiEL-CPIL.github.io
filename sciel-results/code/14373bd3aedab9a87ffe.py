import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_mask = (1 << n) - 1
    deadline = time.time() + 85.0
    
    def find_hc(adj_mask_local, rng):
        for v in range(n):
            if adj_mask_local[v].bit_count() < 2:
                return None
        
        visited_mask = 1
        path = [0]
        
        def dfs(current, depth):
            if depth == n:
                if (adj_mask_local[current] & 1) != 0:
                    return True
                return False
            
            avail = adj_mask_local[current] & (all_mask ^ visited_mask)
            if avail == 0:
                return False
            
            neighbors = []
            temp_avail = avail
            while temp_avail:
                lsb = temp_avail & (-temp_avail)
                nxt = lsb.bit_length() - 1
                neighbors.append(nxt)
                temp_avail ^= lsb
            
            # Heuristic
            neighbors.sort(key=lambda v: (adj_mask_local[v] & (all_mask ^ visited_mask)).bit_count())
            rng.shuffle(neighbors) # Shuffle to break ties randomly
            # Re-sort to maintain heuristic but randomize ties
            # Actually, sort is stable, so if we shuffle first, then sort, ties are random.
            # But we want to sort by degree.
            # So: Shuffle, then Stable Sort by degree.
            
            for nxt in neighbors:
                visited_mask |= (1 << nxt)
                path.append(nxt)
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited_mask ^= (1 << nxt)
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None

    best_cycles = []
    
    while time.time() < deadline:
        current_adj = adj_mask[:]
        cycles = []
        
        for i in range(3):
            rng = random.Random()
            hc = find_hc(current_adj, rng)
            if hc is None:
                break
            cycles.append(hc)
            for j in range(len(hc)):
                u = hc[j]
                v = hc[(j + 1) % len(hc)]
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            
        if len(cycles) == 3:
            # Verify
            valid = True
            used_edges = set()
            for c in cycles:
                if len(c) != n or len(set(c)) != n:
                    valid = False
                    break
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j + 1) % len(c)]
                    edge = (min(u, v), max(u, v))
                    if edge in used_edges:
                        valid = False
                        break
                    used_edges.add(edge)
                    if (adj_mask[u] & (1 << v)) == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return [[int(v) for v in c] for c in cycles]
                
    return [[int(v) for v in c] for c in best_cycles]

if __name__ == '__main__':
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Found {len(result)} cycles in {elapsed:.2f}s")
