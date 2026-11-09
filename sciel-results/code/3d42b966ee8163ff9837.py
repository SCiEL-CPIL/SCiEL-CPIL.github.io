"""Rewritten program: pack distinct multiples of a fixed integer set."""
# EVOLVE-BLOCK-START
import random

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _pack_multiples():
    """
    Uses randomized local search (hill climbing) to minimize the span.
    1. Initialize with random distinct multiples.
    2. Iteratively try to move the current minimum up and the current maximum down.
    3. If stuck, perform random local moves to escape local minima.
    """
    best_sol = None
    best_span = float('inf')
    
    def generate_initial_solution():
        sol = [0] * N
        used = set()
        # Pick random multiples in a moderate range to start
        for i in range(N):
            d = DIVISORS[i]
            found = False
            for _ in range(1000):
                k = random.randint(10, 500)
                m = k * d
                if m not in used:
                    sol[i] = m
                    used.add(m)
                    found = True
                    break
            if not found:
                # Fallback: linear search from a random start
                start_k = random.randint(10, 500)
                k = start_k
                while True:
                    m = k * d
                    if m not in used:
                        sol[i] = m
                        used.add(m)
                        break
                    k += 1
        return sol

    # Run multiple independent local searches
    for _ in range(50):
        sol = generate_initial_solution()
        span = max(sol) - min(sol)
        
        improved = True
        while improved:
            improved = False
            cur_min = min(sol)
            cur_max = max(sol)
            
            # Try to move min up
            min_idx = sol.index(cur_min)
            d_min = DIVISORS[min_idx]
            new_min = cur_min + d_min
            # Only move if it stays below current max to guarantee span reduction
            if new_min < cur_max and new_min not in sol:
                sol[min_idx] = new_min
                new_span = max(sol) - min(sol)
                if new_span < span:
                    span = new_span
                    improved = True
                else:
                    sol[min_idx] = cur_min # Revert
            
            # Try to move max down
            max_idx = sol.index(cur_max)
            d_max = DIVISORS[max_idx]
            new_max = cur_max - d_max
            # Only move if it stays above current min to guarantee span reduction
            if new_max > cur_min and new_max not in sol:
                sol[max_idx] = new_max
                new_span = max(sol) - min(sol)
                if new_span < span:
                    span = new_span
                    improved = True
                else:
                    sol[max_idx] = cur_max # Revert
            
            # If no simple improvement, try random kicks
            if not improved:
                # Try to move a random element to a nearby multiple
                # This helps escape local minima where min/max are blocked
                kick_success = False
                for _ in range(10):
                    idx = random.randint(0, N - 1)
                    d = DIVISORS[idx]
                    old_val = sol[idx]
                    # Try offsets -2, -1, 1, 2
                    for offset in [-2, -1, 1, 2]:
                        new_val = old_val + offset * d
                        if new_val <= 0 or new_val > MAX_VAL:
                            continue
                        if new_val in sol:
                            continue
                        
                        sol[idx] = new_val
                        new_span = max(sol) - min(sol)
                        
                        if new_span < span:
                            span = new_span
                            improved = True
                            kick_success = True
                            break
                        else:
                            sol[idx] = old_val
                    
                    if kick_success:
                        break
                
                # If we still didn't improve, we are in a local minimum
                # We could accept a non-improving move with some probability (simulated annealing)
                # But for simplicity and speed, we stop this run.
                pass
        
        if span < best_span:
            best_span = span
            best_sol = sol[:]
            
    return best_sol, best_span

_optimal_solution = None

def _get_solution():
    global _optimal_solution
    if _optimal_solution is None:
        _optimal_solution = _pack_multiples()
    return _optimal_solution

def construct():
    sol, _ = _get_solution()
    # Ensure we return Python ints in a list
    return [int(x) for x in sol]

# EVOLVE-BLOCK-END
