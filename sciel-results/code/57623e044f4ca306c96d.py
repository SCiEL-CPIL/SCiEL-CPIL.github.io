"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    deadline = start_time + 270.0

    # Precompute primes up to N
    all_primes = _primes(N)
    primes_set = set(all_primes)

    # Phase 1: Top-down removal with better lookahead
    A = set(range(N + 1))
    primes = _primes(N)

    # Precompute residues for each prime to speed up counting
    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
        if p > len(A):
            break

        # Count residues
        counts = [0] * p
        for x in A:
            counts[x % p] += 1

        min_count = min(counts)
        if min_count == 0:
            continue

        candidates = [r for r in range(p) if counts[r] == min_count]
        r_to_remove = candidates[0]

        # Multi-step lookahead: try to maximize future admissibility
        if len(candidates) > 1 and i + 1 < len(primes):
            best_r = candidates[0]
            best_score = -1
            # Look ahead 2 steps if possible
            for r in candidates:
                # Simulate removal
                A_sim = {x for x in A if x % p != r}
                score = len(A_sim)
                
                # Check next prime
                if i + 1 < len(primes):
                    p_next = primes[i + 1]
                    if p_next <= len(A_sim):
                        next_counts = [0] * p_next
                        for x in A_sim:
                            next_counts[x % p_next] += 1
                        min_next = min(next_counts)
                        # Penalize if we create a very unbalanced distribution
                        score -= min_next * 0.1
                        # Bonus if we keep more balance
                        score += sum(1 for c in next_counts if c > 0) * 0.05
                
                if score > best_score:
                    best_score = score
                    best_r = r
            r_to_remove = best_r

        A = {x for x in A if x % p != r_to_remove}
        if not A:
            break

    # Phase 2: Greedy addition with precomputed residue counts
    # Maintain residue counts for each relevant prime
    # For each prime p <= len(A) + 1, track which residues are present
    
    # Build set of relevant primes (those <= len(A) + 1)
    relevant_primes = [p for p in all_primes if p <= len(A) + 1]
    
    # Precompute residue presence for each relevant prime
    # residue_present[p][r] = True if any x in A has x % p == r
    residue_present = {}
    for p in relevant_primes:
        present = [False] * p
        for x in A:
            present[x % p] = True
        residue_present[p] = present
    
    # Count how many residues are missing for each prime
    def count_missing(p):
        return sum(1 for r in range(p) if not residue_present[p][r])
    
    # Candidate list: all numbers not in A
    candidates = [x for x in range(N + 1) if x not in A]
    
    # Score candidates: prefer those that don't fill the last missing residue for any prime
    # A candidate is "safe" if for all relevant primes p, residue_present[p][cand % p] is True
    # (i.e., the residue is already present, so adding doesn't fill a gap)
    
    def is_safe(cand):
        new_size = len(A) + 1
        for p in relevant_primes:
            if p > new_size:
                break
            r = cand % p
            if not residue_present[p][r]:
                # This would fill the last missing residue for prime p
                # Check if there are other missing residues for p
                # If this is the ONLY missing residue, it's unsafe
                missing_count = sum(1 for rr in range(p) if not residue_present[p][rr])
                if missing_count == 1:
                    return False
        return True
    
    # Sort candidates: safe ones first, then by some heuristic
    # For safe candidates, prefer those that are "less constraining"
    # A number is less constraining if it shares residues with many existing elements
    # Actually, let's just try safe candidates in order, then unsafe ones
    
    safe_candidates = []
    unsafe_candidates = []
    
    for cand in candidates:
        if is_safe(cand):
            safe_candidates.append(cand)
        else:
            unsafe_candidates.append(cand)
    
    # Try safe candidates first
    for cand in safe_candidates:
        if time.time() > deadline:
            break
        
        # Add candidate and update residue_present
        A.add(cand)
        new_size = len(A)
        
        # Update residue_present for all relevant primes
        for p in relevant_primes:
            if p <= new_size:
                r = cand % p
                residue_present[p][r] = True
        
        # If new_size increased, we might need to add new relevant primes
        # But we'll handle that in the next iteration by rebuilding if needed
    
    # Try unsafe candidates if time remains
    for cand in unsafe_candidates:
        if time.time() > deadline:
            break
        
        # For unsafe candidates, we need to be more careful
        # Check if adding this candidate would make any prime have all residues present
        new_size = len(A) + 1
        safe_to_add = True
        
        for p in relevant_primes:
            if p > new_size:
                break
            r = cand % p
            if not residue_present[p][r]:
                # This residue is missing. Check if it's the only missing one
                missing = [rr for rr in range(p) if not residue_present[p][rr]]
                if len(missing) == 1:
                    safe_to_add = False
                    break
        
        if safe_to_add:
            A.add(cand)
            new_size = len(A)
            for p in relevant_primes:
                if p <= new_size:
                    r = cand % p
                    residue_present[p][r] = True
    
    return list(A)
# EVOLVE-BLOCK-END
