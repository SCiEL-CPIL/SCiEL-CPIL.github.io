"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    deadline = time.time() + 290  # safety margin
    
    points = set(range(N + 1))
    primes = _primes(N + 1)
    prime_set = set(primes)
    
    # 1. Greedy Removal Phase
    # Iterate through primes in increasing order.
    # If p <= len(points) and all residues covered, remove least populated class.
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(points):
            break
            
        # Check if all residues are covered
        counts = [0] * p
        for x in points:
            counts[x % p] += 1
            
        if min(counts) > 0:
            # All residues covered
            min_class_size = min(counts)
            min_residue = counts.index(min_class_size)
            to_remove = {x for x in points if x % p == min_residue}
            points -= to_remove

    # 2. Expansion Phase
    k = len(points)
    
    # Relevant primes are those p <= k+1
    # We will maintain empty_res for these primes
    empty_res = {}
    
    def compute_empty(p):
        present = set()
        for x in points:
            present.add(x % p)
        return set(range(p)) - present

    # Initialize empty_res for primes <= k+1
    for p in primes:
        if p <= k + 1:
            empty_res[p] = compute_empty(p)
        else:
            break

    forbidden = [False] * (N + 1)
    
    def mark_forbidden(p, r):
        # Mark all x in [0, N] not in points such that x % p == r as forbidden
        for x in range(r, N + 1, p):
            if x not in points:
                forbidden[x] = True

    # Initial marking
    for p, empties in empty_res.items():
        if len(empties) == 1:
            r = next(iter(empties))
            mark_forbidden(p, r)

    # Iterate through candidates
    for x in range(N + 1):
        if time.time() > deadline:
            break
            
        if x in points:
            continue
            
        if forbidden[x]:
            continue
            
        new_k = k + 1
        valid = True
        
        # Check if new_k is a prime and if x violates it
        if new_k in prime_set:
            # Compute empty residues for new_k based on current points (size new_k - 1)
            present = set()
            for y in points:
                present.add(y % new_k)
            empties_new = set(range(new_k)) - present
            
            if len(empties_new) == 1:
                r_new = next(iter(empties_new))
                if x % new_k == r_new:
                    valid = False
            # If len(empties_new) > 1, it's safe to add x for this prime
            # If len(empties_new) == 0, it's already a violation (should not happen if previous steps were correct)
        
        if valid:
            points.add(x)
            k = new_k
            
            # Update empty_res for existing primes
            for p in list(empty_res.keys()):
                if p > k:
                    continue
                r_x = x % p
                if r_x in empty_res[p]:
                    empty_res[p].remove(r_x)
                    if len(empty_res[p]) == 1:
                        r_new = next(iter(empty_res[p]))
                        mark_forbidden(p, r_new)
                    # If len becomes 0, we have a problem, but we checked validity, so this shouldn't happen
                    # Actually, if we removed the last empty residue, it means we filled all residues.
                    # This would mean the set is no longer admissible.
                    # But we only add x if it doesn't fill the last hole.
                    # So if r_x was in empty_res[p], it was the only one?
                    # If len(empty_res[p]) was 1, and we removed it, len becomes 0.
                    # This means x filled the last hole.
                    # But our check `if forbidden[x]` should have caught this?
                    # `forbidden[x]` was set to True if there was a unique empty residue r and x % p == r.
                    # So if x % p == r_x and r_x was the unique empty residue, forbidden[x] would be True.
                    # So we would not add x.
                    # Thus, len(empty_res[p]) should never become 0.
            
            # If new_k is prime, add it to tracking
            if k in prime_set:
                # We need to compute empty_res for this new prime k
                # Note: we already computed `empties_new` in the validity check if k was prime
                # But we didn't store it. Let's recompute or store.
                # For efficiency, we can just recompute, it's O(k).
                empty_res[k] = compute_empty(k)
                if len(empty_res[k]) == 1:
                    r_new = next(iter(empty_res[k]))
                    mark_forbidden(k, r_new)

    return list(points)
# EVOLVE-BLOCK-END
