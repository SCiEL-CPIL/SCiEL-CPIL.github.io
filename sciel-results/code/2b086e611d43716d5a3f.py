import time
import random

def construct():
    start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    N = 180

    # Precompute all reciprocal Schur triples (a, b, c) with 1/a = 1/b + 1/c, a,b,c distinct
    # 1/a = 1/b + 1/c => bc = a(b+c) => (b-a)(c-a) = a^2
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                if d == e:
                    continue
                b1 = a + d
                c1 = a + e
                if b1 <= N and c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    triples.append((a, b1, c1))
                b2 = a + e
                c2 = a + d
                if b2 <= N and c2 <= N and b2 != c2 and b2 != a and c2 != a:
                    if (a, b2, c2) != (a, b1, c1):
                        triples.append((a, b2, c2))

    # Deduplicate and convert to masks
    seen = set()
    unique_triple_masks = []
    conflict_masks = [[] for _ in range(N + 1)]
    
    for t in triples:
        fs = frozenset(t)
        if len(fs) == 3:
            if fs not in seen:
                seen.add(fs)
                a, b, c = sorted(fs)
                # Bit i corresponds to number i+1? No, let's use 1-indexed bits for clarity or 0-indexed.
                # Let's use 0-indexed bits: bit i represents element i+1.
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                unique_triple_masks.append(mask)
                
                # Precompute conflict masks for each element in the triple
                # For element a, the other two are b, c. Conflict mask is bits for b and c.
                mask_bc = (1 << (b - 1)) | (1 << (c - 1))
                conflict_masks[a].append(mask_bc)
                
                mask_ac = (1 << (a - 1)) | (1 << (c - 1))
                conflict_masks[b].append(mask_ac)
                
                mask_ab = (1 << (a - 1)) | (1 << (b - 1))
                conflict_masks[c].append(mask_ab)

    def is_valid(mask):
        for t_mask in unique_triple_masks:
            if (mask & t_mask) == t_mask:
                return False
        return True

    def fix_mask(mask):
        """Remove minimal elements to make mask valid."""
        changed = True
        while changed:
            changed = False
            # Find a violating triple
            for t_mask in unique_triple_masks:
                if (mask & t_mask) == t_mask:
                    # Remove one element from this triple
                    # Heuristic: remove the element with the highest value? Or lowest?
                    # Or remove the element that appears in the most other violating triples?
                    # Simple: remove the first element (lowest value) from the triple
                    # To find which element is which, we need to check bits.
                    # a, b, c are 1-indexed.
                    # Let's just iterate through bits in t_mask
                    bits = []
                    for i in range(N):
                        if (t_mask >> i) & 1:
                            bits.append(i)
                    # bits contains 0-indexed indices of the 3 elements
                    # Pick one to remove. Let's pick the one that maximizes removal impact?
                    # For now, pick the first one (smallest element)
                    remove_idx = bits[0]
                    mask &= ~(1 << remove_idx)
                    changed = True
                    break
        return mask

    def can_add(mask, x):
        """Check if adding element x (1-indexed) to mask creates a violation."""
        # x is 1-indexed, so bit index is x-1
        # We need to check if any conflict mask for x is fully contained in mask
        # conflict_masks[x] contains masks of the *other two* elements
        for cm in conflict_masks[x]:
            if (mask & cm) == cm:
                return False
        return True

    # Strategy 1: Start from full set and greedily remove
    S_full_mask = (1 << N) - 1
    S_greedy_mask = fix_mask(S_full_mask)
    best_mask = S_greedy_mask
    best_size = bin(best_mask).count('1')

    # Strategy 2: Start from upper interval
    # {73, ..., 180}
    # Bits 72 to 179
    S_interval_mask = 0
    for i in range(72, N):
        S_interval_mask |= (1 << i)
    
    # Check if interval is valid (it should be, but let's be sure)
    if not is_valid(S_interval_mask):
        S_interval_mask = fix_mask(S_interval_mask)
        
    if bin(S_interval_mask).count('1') > best_size:
        best_mask = S_interval_mask
        best_size = bin(best_mask).count('1')

    # Hill climbing: try adding each excluded element
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        excluded = [i for i in range(1, N + 1) if not (best_mask & (1 << (i - 1)))]
        if not excluded:
            break
        improved = False
        for x in excluded:
            if time.time() > DEADLINE:
                break
            if can_add(best_mask, x):
                new_mask = best_mask | (1 << (x - 1))
                # It's possible that adding x is valid locally but we might want to try to add more?
                # No, hill climbing adds one at a time.
                # Since can_add checks for immediate violations, new_mask is valid.
                if bin(new_mask).count('1') > best_size:
                    best_mask = new_mask
                    best_size = bin(best_mask).count('1')
                    improved = True
        if not improved:
            break

    # Simulated annealing / random search
    rng = random.Random(42)
    for iteration in range(5000):
        if time.time() > DEADLINE:
            break
        if best_size >= N:
            break
        excluded = [i for i in range(1, N + 1) if not (best_mask & (1 << (i - 1)))]
        if not excluded:
            break
        x = rng.choice(excluded)
        if can_add(best_mask, x):
            new_mask = best_mask | (1 << (x - 1))
            if bin(new_mask).count('1') > best_size:
                best_mask = new_mask
                best_size = bin(best_mask).count('1')

    # Try different starting points with random initialization
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        # Random subset of size ~N*0.7
        random_indices = rng.sample(range(N), int(N * 0.7))
        random_mask = 0
        for idx in random_indices:
            random_mask |= (1 << idx)
        
        fixed_mask = fix_mask(random_mask)
        
        # Hill climb from this point
        for _ in range(20):
            if time.time() > DEADLINE:
                break
            excluded = [i for i in range(1, N + 1) if not (fixed_mask & (1 << (i - 1)))]
            if not excluded:
                break
            improved = False
            for x in excluded:
                if time.time() > DEADLINE:
                    break
                if can_add(fixed_mask, x):
                    new_mask = fixed_mask | (1 << (x - 1))
                    if bin(new_mask).count('1') > bin(fixed_mask).count('1'):
                        fixed_mask = new_mask
                        improved = True
            if not improved:
                break
        
        if bin(fixed_mask).count('1') > best_size:
            best_mask = fixed_mask
            best_size = bin(best_mask).count('1')

    # Final verification and cleanup
    # Convert mask to list
    final = []
    for i in range(N):
        if best_mask & (1 << i):
            final.append(i + 1)

    if not final:
        final = list(range(73, 180))

    # Double-check validity using the original set logic to be absolutely safe
    in_final = set(final)
    # Re-construct triples list for final check if needed, or just use masks
    # Using masks is faster
    final_mask = 0
    for x in final:
        final_mask |= (1 << (x - 1))
        
    # If invalid, remove elements greedily
    if not is_valid(final_mask):
        # This should not happen if our logic is correct, but just in case
        temp_mask = final_mask
        temp_mask = fix_mask(temp_mask)
        final = []
        for i in range(N):
            if temp_mask & (1 << i):
                final.append(i + 1)

    if not final:
        final = [1]

    return final
