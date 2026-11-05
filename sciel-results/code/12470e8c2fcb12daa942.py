import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = 45.0  # Safety margin
    
    # Precompute all 3-APs as frozensets of pairs for fast conflict checking
    # For each value v, precompute which pairs (a, b) in [1,n] would form a 3-AP with v
    # Actually, let's precompute for each v, the set of "conflicting pairs"
    # A pair (a, b) conflicts with v if {a, b, v} forms a 3-AP
    
    # Better approach: for each v, precompute the set of values that conflict with v
    # when added to a set S. v conflicts with S if there exists a in S such that:
    #   1. 2*v - a is in S (v is middle)
    #   2. (a + v) % 2 == 0 and (a+v)//2 is in S (v is end/start)
    
    # For speed, let's precompute for each v, the list of "partner" values
    # such that adding v to S is invalid if any partner is also in S.
    # Actually, the conflict depends on two elements in S, so it's pairwise.
    
    # Let's use a different approach: precompute for each v, all pairs (a, b) 
    # where {a, b, v} is a 3-AP. Then v is invalid to add if any such pair 
    # is entirely contained in S.
    
    # Precompute: for each v in 1..n, list of frozenset({a, b}) such that 
    # {a, b, v} is a non-constant 3-AP
    ap_pairs = [set() for _ in range(n + 1)]
    for v in range(1, n + 1):
        for a in range(1, n + 1):
            if a == v:
                continue
            # v is middle: a, v, c where c = 2v - a
            c = 2 * v - a
            if 1 <= c <= n and c != v:
                ap_pairs[v].add(frozenset((a, c)))
            # v is end or start: a, b, v or v, b, a where b = (a+v)/2
            if (a + v) % 2 == 0:
                b = (a + v) // 2
                if 1 <= b <= n and b != v:
                    ap_pairs[v].add(frozenset((a, b)))
    
    def is_valid_addition(S, v):
        """Check if v can be added to S without creating a 3-AP."""
        for pair in ap_pairs[v]:
            # Check if both elements of pair are in S
            # pair is a frozenset of size 1 or 2
            # If size 1, it's a degenerate case (shouldn't happen for non-constant APs)
            # If size 2, check both
            if len(pair) == 2:
                a, b = tuple(pair)
                if a in S and b in S:
                    return False
        return True

    def get_score(S):
        return sum(1.0 / x for x in S)

    def greedy_fill(S, order):
        """Add elements from order to S if valid."""
        for x in order:
            if x not in S and is_valid_addition(S, x):
                S.add(x)
        return S

    def local_search(current_S, current_score, time_limit):
        """Perform 1-swap local search until no improvement or time limit."""
        no_improvement_count = 0
        while time.time() - start_time < time_limit:
            improved = False
            in_S = list(current_S)
            not_in_S = [x for x in range(1, n + 1) if x not in current_S]
            
            # Sort by potential gain: prioritize small v (high weight) replacing large u (low weight)
            # Try all swaps
            for u in in_S:
                for v in not_in_S:
                    if v >= u:
                        continue
                    
                    # Check if swap is valid
                    test_S = current_S.copy()
                    test_S.remove(u)
                    
                    if is_valid_addition(test_S, v):
                        new_score = current_score - 1.0 / u + 1.0 / v
                        if new_score > current_score + 1e-9:
                            current_S = test_S
                            current_S.add(v)
                            current_score = new_score
                            improved = True
                            no_improvement_count = 0
                            break
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > 3:
                    break
        
        return current_S, current_score

    def perturb_and_refill(current_S, num_removals, time_limit):
        """Remove num_removals random elements, then refill with shuffled order."""
        if len(current_S) <= num_removals + 1:
            return current_S, get_score(current_S)
        
        perturbed_S = set(current_S)
        to_remove = random.sample(list(perturbed_S), num_removals)
        for u in to_remove:
            perturbed_S.remove(u)
        
        # Refill with shuffled order for diversity
        remaining = [x for x in range(1, n + 1) if x not in perturbed_S]
        random.shuffle(remaining)
        greedy_fill(perturbed_S, remaining)
        
        return perturbed_S, get_score(perturbed_S)

    best_S = None
    best_score = 0.0

    # Multiple randomized greedy initializations
    num_initializations = 10
    per_init_time = deadline / (num_initializations + 5)  # Reserve time for local search
    
    for init_idx in range(num_initializations):
        if time.time() - start_time > deadline - 5:
            break
        
        # Create a randomized order
        order = list(range(1, n + 1))
        # Mix: keep small numbers somewhat prioritized but shuffle
        # Use a biased shuffle: smaller numbers have higher probability of being early
        random.shuffle(order)
        # Sort by a random key that favors small numbers
        order.sort(key=lambda x: random.random() ** 2 * x)
        
        S = set()
        greedy_fill(S, order)
        
        if get_score(S) > best_score:
            best_S = set(S)
            best_score = get_score(best_S)
        
        # Local search on this initialization
        if time.time() - start_time < deadline - 10:
            S, score = local_search(S, get_score(S), deadline - 10)
            if score > best_score:
                best_S = set(S)
                best_score = score

    # Continue local search on best solution
    if best_S is not None and time.time() - start_time < deadline - 5:
        best_S, best_score = local_search(best_S, best_score, deadline - 5)
    
    # Perturbation phase: try to escape local optima
    if best_S is not None and time.time() - start_time < deadline - 2:
        current_S = set(best_S)
        current_score = best_score
        no_improvement = 0
        
        while time.time() - start_time < deadline - 2:
            # Try various perturbation sizes
            num_removals = random.randint(1, 5)
            perturbed_S, perturbed_score = perturb_and_refill(current_S, num_removals, deadline - 2)
            
            # Local search on perturbed solution
            if time.time() - start_time < deadline - 3:
                perturbed_S, perturbed_score = local_search(perturbed_S, perturbed_score, deadline - 3)
            
            if perturbed_score > current_score + 1e-9:
                current_S = perturbed_S
                current_score = perturbed_score
                no_improvement = 0
                if current_score > best_score:
                    best_S = set(current_S)
                    best_score = current_score
            else:
                no_improvement += 1
                if no_improvement > 20:
                    break

    if best_S is None:
        # Fallback: natural order greedy
        S = set()
        for x in range(1, n + 1):
            if is_valid_addition(S, x):
                S.add(x)
        best_S = S
        best_score = get_score(best_S)

    return list(best_S)
