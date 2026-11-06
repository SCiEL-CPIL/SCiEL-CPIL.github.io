"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256

    # Precompute conflict table: conflicts[x] = set of y such that x and y cannot both be in a 3-AP-free set
    # x and y conflict if there exists z such that {x,y,z} forms a nonconstant 3-AP
    # For pair (x,y), they conflict if:
    #   - x is middle: y = 2x - z for some z, i.e., y and x have same parity and |y-x| even... 
    #   Actually: three numbers a,b,c form a 3-AP if a+c=2b.
    #   For a pair (x,y), they are in conflict if there exists a third element z in [1,N]
    #   such that {x,y,z} is a 3-AP. This happens when:
    #   - x+y is even and (x+y)//2 is in [1,N] and (x+y)//2 != x and (x+y)//2 != y
    #   - or y+x is even... wait, let me think more carefully.
    #   
    #   Three distinct elements form a 3-AP if one is the midpoint of the other two.
    #   For pair (x,y) with x != y:
    #   - If x is the midpoint: need z such that x = (y+z)/2, i.e., z = 2x-y, and z in [1,N], z != x, z != y
    #   - If y is the midpoint: need z such that y = (x+z)/2, i.e., z = 2y-x, and z in [1,N], z != x, z != y
    #   - If some z is the midpoint: z = (x+y)/2, need x+y even, z in [1,N], z != x, z != y
    #   
    #   So x and y conflict if any of these conditions hold.

    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            # Check if x and y can coexist in a 3-AP-free set
            # They conflict if adding both would create a 3-AP with some third element
            conflict = False
            # x is midpoint: z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                conflict = True
            # y is midpoint: z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                conflict = True
            # z is midpoint: z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    conflict = True
            if conflict:
                conflicts[x].add(y)
                conflicts[y].add(x)

    def is_valid_addition(selected_set, x):
        """Check if x can be added to selected_set without creating a 3-AP."""
        # x conflicts with any element in selected_set
        for a in selected_set:
            if a in conflicts[x]:
                return False
        return True

    def get_score_float(selected):
        """Compute float sum of reciprocals."""
        return sum(1.0 / x for x in selected)

    def get_score_fraction(selected):
        """Compute exact Fraction sum of reciprocals."""
        total = Fraction(0, 1)
        for x in selected:
            total += Fraction(1, x)
        return total

    # Multiple greedy initializations
    def greedy(order):
        """Greedy construction with given order of elements."""
        selected = set()
        for x in order:
            if is_valid_addition(selected, x):
                selected.add(x)
        return selected

    # Try several different orderings
    best_set = None
    best_score = -1.0

    # Ordering 1: smallest first (maximize reciprocal weight)
    s1 = greedy(range(1, N + 1))
    sc1 = get_score_float(s1)
    if best_set is None or sc1 > best_score:
        best_set, best_score = s1, sc1

    # Ordering 2: random-like permutation using a simple LCG
    import random
    rng = random.Random(42)
    order2 = list(range(1, N + 1))
    rng.shuffle(order2)
    s2 = greedy(order2)
    sc2 = get_score_float(s2)
    if sc2 > best_score:
        best_set, best_score = s2, sc2

    # Ordering 3: another random
    rng2 = random.Random(123)
    order3 = list(range(1, N + 1))
    rng2.shuffle(order3)
    s3 = greedy(order3)
    sc3 = get_score_float(s3)
    if sc3 > best_score:
        best_set, best_score = s3, sc3

    # Local search with 1-swap and 2-swap moves
    def local_search(selected_set, time_limit):
        """Perform local search with 1-swap and 2-swap moves."""
        current_set = set(selected_set)
        current_score = get_score_float(current_set)
        improved = True
        stagnation = 0
        
        while improved and time.time() < time_limit:
            improved = False
            found_any = False
            
            # Try 1-swap moves: remove one, add one
            for remove in list(current_set):
                if time.time() > time_limit:
                    break
                # Only try adding elements that are not already in the set
                for add in range(1, N + 1):
                    if add in current_set:
                        continue
                    # Quick check: does 'add' conflict with any element in (current_set - {remove})?
                    new_set_without_remove = current_set - {remove}
                    valid = True
                    for a in new_set_without_remove:
                        if a in conflicts[add]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Compute new score
                    new_score = current_score - 1.0 / remove + 1.0 / add
                    if new_score > current_score + 1e-12:
                        current_set = new_set_without_remove | {add}
                        current_score = new_score
                        improved = True
                        found_any = True
                        break
                if improved:
                    break
            
            if not found_any:
                # Try 2-swap moves: remove two, add two
                elements = list(current_set)
                for i in range(len(elements)):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, len(elements)):
                        if time.time() > time_limit:
                            break
                        r1, r2 = elements[i], elements[j]
                        base_set = current_set - {r1, r2}
                        base_score = current_score - 1.0 / r1 - 1.0 / r2
                        
                        for add1 in range(1, N + 1):
                            if add1 in current_set:
                                continue
                            # Check add1 against base_set
                            valid1 = True
                            for a in base_set:
                                if a in conflicts[add1]:
                                    valid1 = False
                                    break
                            if not valid1:
                                continue
                            
                            for add2 in range(add1 + 1, N + 1):
                                if add2 in current_set:
                                    continue
                                # Check add2 against (base_set + {add1})
                                valid2 = True
                                for a in base_set:
                                    if a in conflicts[add2]:
                                        valid2 = False
                                        break
                                if valid2 and add1 in conflicts[add2]:
                                    valid2 = False
                                if not valid2:
                                    continue
                                
                                new_score = base_score + 1.0 / add1 + 1.0 / add2
                                if new_score > current_score + 1e-12:
                                    current_set = base_set | {add1, add2}
                                    current_score = new_score
                                    improved = True
                                    found_any = True
                                    break
                            if found_any:
                                break
                        if found_any:
                            break
                    if found_any:
                        break
            
            if not found_any:
                stagnation += 1
                if stagnation > 3:
                    break
            else:
                stagnation = 0
        
        return current_set, current_score

    # Run local search with a time budget
    search_deadline = deadline - 5.0  # leave 5 seconds for final verification
    best_set, best_score = local_search(best_set, search_deadline)

    # Final verification
    if time.time() < deadline:
        # Verify the result is valid
        valid = True
        for x in best_set:
            for a in best_set:
                if a == x:
                    continue
                if a in conflicts[x]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to greedy result
            best_set = set()
            for x in range(1, N + 1):
                if is_valid_addition(best_set, x):
                    best_set.add(x)

    return sorted(best_set)
# EVOLVE-BLOCK-END
