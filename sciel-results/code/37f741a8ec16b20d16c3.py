"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    # 1. Generate all 5-smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth.append(x)
    
    # Sort them
    smooth.sort()
    n_smooth = len(smooth)
    
    # 2. Find minimal number of terms for each target N in 1..128
    results = []
    
    # Precompute pairs and triples for speed? 
    # Or just loop.
    
    for n in range(1, 129):
        # k=1
        found = False
        for x in smooth:
            if x == n:
                results.append(1)
                found = True
                break
            if x > n:
                break
        if found: continue
        
        # k=2
        found = False
        for i in range(n_smooth):
            a = smooth[i]
            if a > n: break
            for j in range(i+1, n_smooth):
                b = smooth[j]
                s = a + b
                if s > n: break
                if s == n:
                    if b % a != 0 and a % b != 0:
                        results.append(2)
                        found = True
                        break
            if found: break
        if found: continue
        
        # k=3
        found = False
        for i in range(n_smooth):
            a = smooth[i]
            if a > n: break
            for j in range(i+1, n_smooth):
                b = smooth[j]
                if a + b > n: break
                for k in range(j+1, n_smooth):
                    c = smooth[k]
                    s = a + b + c
                    if s > n: break
                    if s == n:
                        if (c % a != 0 and a % c != 0 and
                            c % b != 0 and b % c != 0 and
                            b % a != 0 and a % b != 0):
                            results.append(3)
                            found = True
                            break
                if found: break
            if found: break
        if found: continue
        
        # k>=4: Use DFS
        def dfs_min(rem, start, depth):
            if rem == 0:
                return depth
            if rem < 0:
                return float('inf')
            
            best = float('inf')
            for i in range(start, n_smooth):
                x = smooth[i]
                if x > rem:
                    break
                # Check antichain with previously chosen?
                # We need to pass chosen numbers.
                # To keep it simple and fast for rare cases, we pass a list.
                # But this function is redefined in the loop, which is inefficient.
                # Let's move the DFS helper outside or make it a closure with access to smooth.
                
                # For now, inline logic is hard. Let's use a nested function that captures 'smooth' and 'n_smooth'.
                # But we need 'chosen' to check antichain.
                
                pass
        
        # Let's implement a proper DFS for k>=4
        # Since it's rare, we can afford a slightly heavier setup.
        
        def solve_k4_plus(target):
            # Returns min number of terms >= 4
            # We know k=1,2,3 failed.
            
            def dfs(rem, start, chosen):
                if rem == 0:
                    return len(chosen)
                if rem < 0:
                    return float('inf')
                
                best = float('inf')
                for i in range(start, n_smooth):
                    x = smooth[i]
                    if x > rem:
                        break
                    # Antichain check
                    ok = True
                    for y in chosen:
                        if x % y == 0 or y % x == 0:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    r = dfs(rem - x, i + 1, chosen + [x])
                    if r < best:
                        best = r
                return best
            
            return dfs(target, 0, [])
        
        min_k = solve_k4_plus(n)
        results.append(min_k)
        
    return results
# EVOLVE-BLOCK-END
