import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    # Greedy: try to assign residues
    # Order: try moduli with fewer constraints first (fewer shared factors)
    # Sort by number of divisors (ascending) to handle "easy" ones first
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    # Try different orderings
    best_result = []
    best_count = 0
    
    # Greedy assignment
    def try_greedy(order):
        assigned = {}  # modulus -> residue
        for m in order:
            if time.time() > deadline:
                break
            found = False
            for r in range(m):
                valid = True
                for n, rn in assigned.items():
                    g = math.gcd(m, n)
                    if r % g == rn % g:
                        valid = False
                        break
                if valid:
                    assigned[m] = r
                    found = True
                    break
            if not found:
                # Try all residues more carefully
                pass
        return assigned
    
    # Try ascending order
    order_asc = sorted(moduli, key=lambda x: (num_divisors(x), x))
    result_asc = try_greedy(order_asc)
    best_count = max(best_count, len(result_asc))
    if len(result_asc) > len(best_result):
        best_result = [[r, m] for m, r in result_asc.items()]
    
    # Try descending order
    order_desc = sorted(moduli, key=lambda x: (-num_divisors(x), x))
    result_desc = try_greedy(order_desc)
    best_count = max(best_count, len(result_desc))
    if len(result_desc) > len(best_result):
        best_result = [[r, m] for m, r in result_desc.items()]
    
    # Try random orders
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        order = moduli[:]
        random.shuffle(order)
        result = try_greedy(order)
        if len(result) > len(best_result):
            best_result = [[r, m] for m, r in result.items()]
            best_count = len(result)
    
    # Local search: try to add more moduli
    if time.time() < deadline - 5:
        # Try to add unassigned moduli
        assigned_set = set(m for m, r in [(m, r) for m, r in [(pair[1], pair[0]) for pair in best_result]])
        # Rebuild from best_result
        assigned = {}
        for r, m in best_result:
            assigned[m] = r
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            unassigned = [m for m in moduli if m not in assigned]
            for m in unassigned:
                if time.time() > deadline - 5:
                    break
                for r in range(m):
                    valid = True
                    for n, rn in assigned.items():
                        g = math.gcd(m, n)
                        if r % g == rn % g:
                            valid = False
                            break
                    if valid:
                        assigned[m] = r
                        improved = True
                        break
    
    # Build final result
    result = [[r, m] for m, r in assigned.items()]
    if not result:
        # Fallback
        result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return result
