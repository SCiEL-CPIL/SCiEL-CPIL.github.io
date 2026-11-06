import time
import sys

# Increase recursion limit just in case, though depth is max 127
sys.setrecursionlimit(2000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    # Precompute primes in [2, 128]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [i for i in range(2, 129) if is_prime(i)]
    prime_set = set(primes)

    best_seq = []
    best_len = 0
    timeout_occurred = False

    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Initial greedy pass to get a good starting point
    # Try greedy with primes first
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        # Check validity
        new_tails = []
        prod = x
        for a in reversed(greedy_seq):
            prod *= a
            new_tails.append(prod)
        new_tails.append(x) # x itself
        
        # Note: The loop above adds suffixes ending at x.
        # Wait, the loop `for a in reversed(greedy_seq)` generates:
        # a_last * x, a_last-1 * a_last * x, ...
        # We also need x itself.
        
        # Let's re-implement validity check clearly
        def is_valid_addition(seq, x, products_set):
            # Generate new products ending at x
            new_prods = []
            prod = x
            new_prods.append(prod)
            for a in reversed(seq):
                prod *= a
                new_prods.append(prod)
            
            # Check collision
            for p in new_prods:
                if p in products_set:
                    return False, None
            
            return True, new_prods

    # Greedy construction
    for x in range(2, 129):
        if time.time() > deadline:
            break
        valid, new_tails = is_valid_addition(greedy_seq, x, greedy_prods)
        if valid:
            greedy_seq.append(x)
            greedy_prods.update(new_tails)
    
    update_best(greedy_seq)

    # If we have time, do IDDFS to improve
    if time.time() < deadline - 1.0:
        target_len = best_len + 1
        
        def dfs(idx, seq, prods, target_len):
            nonlocal timeout_occurred
            if timeout_occurred:
                return False
            
            if time.time() > deadline:
                timeout_occurred = True
                return False

            # Pruning: Can we reach target_len?
            if len(seq) + (128 - idx + 1) < target_len:
                return False
            
            if len(seq) == target_len:
                update_best(seq)
                return True
            
            if idx > 128:
                return False

            x = idx
            
            # Heuristic: Primes are safer, so try including them first.
            # Composites are riskier, so try excluding them first (to keep the sequence "clean" for primes).
            # However, we want to ADD composites to increase length.
            # So for composites, we might want to try including them if it helps reach the target.
            # But if we include a composite and it blocks 5 primes, we lose.
            # So, try Exclude first for composites?
            # If we exclude all composites, we get primes.
            # Then we backtrack and try including composites.
            
            # Let's try this order:
            # 1. If prime: Try Include, then Exclude.
            # 2. If composite: Try Exclude, then Include.
            
            if x in prime_set:
                # Try Include
                valid, new_tails = is_valid_addition(seq, x, prods)
                if valid:
                    new_seq = seq + [x]
                    new_prods = prods.union(new_tails)
                    if dfs(idx + 1, new_seq, new_prods, target_len):
                        return True
                
                # Try Exclude
                if dfs(idx + 1, seq, prods, target_len):
                    return True
            else:
                # Try Exclude first
                if dfs(idx + 1, seq, prods, target_len):
                    return True
                
                # Try Include
                valid, new_tails = is_valid_addition(seq, x, prods)
                if valid:
                    new_seq = seq + [x]
                    new_prods = prods.union(new_tails)
                    if dfs(idx + 1, new_seq, new_prods, target_len):
                        return True

            return False

        # Iterate target lengths
        while time.time() < deadline - 1.0:
            if timeout_occurred:
                break
            
            # Optional: Add a small delay or check time more frequently
            if dfs(2, [], set(), target_len):
                target_len += 1
            else:
                break

    return best_seq
