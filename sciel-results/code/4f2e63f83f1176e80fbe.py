import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Candidate moduli: Multiples of 4
    moduli = list(range(4, 129, 4))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    def is_violated(i, j, r):
        return (r[i] - r[j]) % gcds[i][j] == 0
        
    def count_violations(r):
        count = 0
        for i in range(n):
            ri = r[i]
            for j in range(i+1, n):
                if (ri - r[j]) % gcds[i][j] == 0:
                    count += 1
        return count
        
    def local_search():
        r = [random.randint(0, moduli[i]-1) for i in range(n)]
        cur_cost = count_violations(r)
        
        if cur_cost == 0:
            return r
            
        best_r = r[:]
        best_cost = cur_cost
        
        # Simple hill climbing with random restarts inside
        for _ in range(20000):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_val = r[i]
            # Try a few new values
            for _ in range(3):
                new_val = random.randint(0, moduli[i]-1)
                if new_val == old_val:
                    continue
                
                delta = 0
                for j in range(n):
                    if i == j:
                        continue
                    old_v = 1 if (old_val - r[j]) % gcds[i][j] == 0 else 0
                    new_v = 1 if (new_val - r[j]) % gcds[i][j] == 0 else 0
                    delta += new_v - old_v
                    
                if delta < 0:
                    r[i] = new_val
                    cur_cost += delta
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_r = r[:]
                    if cur_cost == 0:
                        return r
                    break # Accept first improving move
                elif delta == 0 and random.random() < 0.2:
                    r[i] = new_val
                    break
                    
        return best_r if best_cost == 0 else None

    # Try to find a solution
    solution = None
    while time.time() < deadline - 2:
        solution = local_search()
        if solution:
            break
            
    if solution:
        # Verify
        if count_violations(solution) == 0:
            return [[solution[i], moduli[i]] for i in range(n)]
            
    # Fallback to seed
    return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
