"""Max-SAT style simulated annealing for reciprocal Schur triple avoidance."""
import time
import random
import math

# Deadline guard: finish well before 90s wall-clock
_DEADLINE = 85.0

def _precompute_triples():
    """
    Find all triples (a, b, c) of distinct integers in [1, 180] such that
    1/a = 1/b + 1/c.
    
    Without loss of generality, assume a < b < c (since the equation is symmetric in b,c,
    and 1/a > 1/b and 1/a > 1/c implies a < b and a < c).
    
    1/a = 1/b + 1/c = (b+c)/(bc)
    => bc = a(b+c)
    => bc - ab - ac = 0
    => bc - ab - ac + a^2 = a^2
    => (b-a)(c-a) = a^2
    
    So for each a, we need factor pairs (d, e) of a^2 such that d*e = a^2,
    then b = a + d, c = a + e, and we need b != c (i.e., d != e) and b, c in [1, 180].
    Also need all three distinct: a != b, a != c (guaranteed since d,e > 0), b != c (d != e).
    """
    triples = []
    for a in range(1, 181):
        a2 = a * a
        # Find all divisors d of a2 where d < e (i.e., d < a2/d) to ensure b != c
        # Actually we need d*e = a2, d != e, and b=a+d <= 180, c=a+e <= 180
        # Iterate d from 1 to sqrt(a2)
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:
                continue  # b would equal c
            b = a + d
            c = a + e
            if b > 180 or c > 180:
                continue
            if b < 1 or c < 1:
                continue
            # Ensure distinctness: a, b, c all distinct
            if a == b or a == c or b == c:
                continue
            triples.append((a, b, c))
    return triples

def _is_valid(set_bits, triples):
    """Check if no triple has all three elements in the set."""
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            return False
    return True

def _count_violations(set_bits, triples):
    """Count number of violated triples."""
    count = 0
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            count += 1
    return count

def construct():
    start_time = time.time()
    
    triples = _precompute_triples()
    n = 180
    
    # Greedy initialization: start with all elements, remove those causing violations
    # Better: use a greedy approach to build a large valid set
    def greedy_init():
        """Greedy: process elements in order, add if no violation with already-added."""
        # Sort elements by how many triples they're in (ascending) - add "less constrained" first
        triple_count = [0] * (n + 1)
        for (a, b, c) in triples:
            triple_count[a] += 1
            triple_count[b] += 1
            triple_count[c] += 1
        
        # Try multiple orderings
        best_set = set()
        for seed in range(20):
            random.seed(seed)
            order = list(range(1, n + 1))
            # Sort by triple count, with random tie-breaking
            random.shuffle(order)
            order.sort(key=lambda x: (triple_count[x], random.random()))
            
            current_set = set()
            set_bits = [False] * (n + 1)
            for x in order:
                # Check if adding x creates any violation
                can_add = True
                for (a, b, c) in triples:
                    if x in (a, b, c):
                        other_two = [v for v in (a, b, c) if v != x]
                        if other_two[0] in current_set and other_two[1] in current_set:
                            can_add = False
                            break
                if can_add:
                    current_set.add(x)
                    set_bits[x] = True
            
            if len(current_set) > len(best_set):
                best_set = current_set
        
        return best_set
    
    # Simulated annealing
    def simulated_annealing(initial_set, time_budget):
        """
        SA on the set: moves are add/remove elements.
        Objective: maximize |set| subject to no violated triples.
        We use a penalty approach: score = |set| - lambda * violations
        """
        set_bits = [False] * (n + 1)
        current_set = set()
        for x in initial_set:
            set_bits[x] = True
            current_set.add(x)
        
        current_size = len(current_set)
        current_violations = _count_violations(set_bits, triples)
        
        # Penalty weight
        lambda_pen = 1000  # Large penalty to strongly discourage violations
        
        def score(s_bits, s_size, s_viol):
            return s_size - lambda_pen * s_viol
        
        current_score = score(set_bits, current_size, current_violations)
        best_set = set(current_set)
        best_size = current_size
        best_score = current_score
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.01
        cooling = 0.9995
        
        T = T_start
        steps = 0
        max_steps = 5000000  # Cap on total steps
        
        while T > T_end and steps < max_steps:
            if time.time() - start_time > _DEADLINE:
                break
            steps += 1
            
            # Choose a random move: add or remove a random element
            x = random.randint(1, n)
            
            if set_bits[x]:
                # Try removing x
                set_bits[x] = False
                new_size = current_size - 1
                # Recount violations (expensive, but we can optimize)
                # For now, recount
                new_violations = _count_violations(set_bits, triples)
                new_score = score(set_bits, new_size, new_violations)
                
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_size = new_size
                    current_violations = new_violations
                    current_score = new_score
                    current_set.discard(x)
                    if new_violations == 0 and new_size > best_size:
                        best_size = new_size
                        best_score = new_score
                        best_set = set(current_set)
                else:
                    set_bits[x] = True  # revert
            else:
                # Try adding x
                set_bits[x] = True
                new_size = current_size + 1
                new_violations = _count_violations(set_bits, triples)
                new_score = score(set_bits, new_size, new_violations)
                
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_size = new_size
                    current_violations = new_violations
                    current_score = new_score
                    current_set.add(x)
                    if new_violations == 0 and new_size > best_size:
                        best_size = new_size
                        best_score = new_score
                        best_set = set(current_set)
                else:
                    set_bits[x] = False  # revert
            
            T *= cooling
        
        return best_set
    
    # Run greedy init
    greedy_set = greedy_init()
    
    # Run SA from greedy init
    sa_set = simulated_annealing(greedy_set, _DEADLINE - (time.time() - start_time))
    
    # Try a few more SA restarts with different initial sets
    best = sa_set
    for restart in range(3):
        if time.time() - start_time > _DEADLINE - 5:
            break
        # Different greedy with different randomization
        random.seed(100 + restart)
        alt_init = greedy_init()
        sa_result = simulated_annealing(alt_init, max(0, _DEADLINE - (time.time() - start_time) - 5))
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final validation
    set_bits = [False] * (n + 1)
    for x in best:
        set_bits[x] = True
    if not _is_valid(set_bits, triples):
        # Fallback to greedy
        best = greedy_set
        set_bits = [False] * (n + 1)
        for x in best:
            set_bits[x] = True
    
    return sorted(best)
