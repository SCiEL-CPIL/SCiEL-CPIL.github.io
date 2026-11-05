import time
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Split denominators into two halves
    nA = 23
    nB = n - nA
    
    valsA = [1.0 / d for d in denoms[:nA]]
    valsB = [1.0 / d for d in denoms[nA:]]
    
    # 1. Generate all subset sums for A (as floats)
    sizeA = 1 << nA
    sumsA = [0.0] * sizeA
    masksA = list(range(sizeA)) # We can just use the index as mask if we generate in order, but we need to sort by sum
    # Actually, we need to associate sum with mask.
    # Let's generate sumsA in a way that we know the mask.
    # The index in the array IS the mask if we use standard binary counting generation?
    # No, standard generation:
    # sumsA[0] = 0
    # for i in range(nA):
    #   bit = 1 << i
    #   for mask in range(sizeA):
    #     if mask & bit:
    #       sumsA[mask] = sumsA[mask ^ bit] + valsA[i]
    # This is O(N * 2^N) which is 23 * 8M = 184M ops. Might be slow in Python.
    
    # Faster generation:
    sumsA = [0.0] * sizeA
    for i in range(nA):
        bit = 1 << i
        val = valsA[i]
        # Iterate over masks that have bit i set
        # Or simply:
        for mask in range(bit):
            # This pattern is tricky. Let's use the standard doubling.
            pass
            
    # Standard efficient generation:
    sumsA = [0.0] * sizeA
    for i in range(nA):
        val = valsA[i]
        bit = 1 << i
        # For all masks, if the bit is set, sum = sum_without_bit + val
        # We can do this in place if we are careful, or just do two loops.
        # Actually, the previous method `sumsA[mask] = sumsA[mask ^ bit] + val` 
        # requires `sumsA[mask ^ bit]` to be already computed.
        # If we iterate `mask` from 0 to sizeA-1, `mask ^ bit` is smaller than `mask` 
        # only if the bit is the highest set bit? No.
        # Example: mask=3 (11), bit=2 (10). mask^bit = 1 (01). 1 < 3. OK.
        # Example: mask=2 (10), bit=2 (10). mask^bit = 0 (00). 0 < 2. OK.
        # So if we iterate i from 0 to nA-1, and for each i, we update all masks?
        # No, we must ensure that when we process bit i, the sums for lower bits are already final.
        # This works if we process bits in increasing order of significance?
        # Let's just use the simple O(N*2^N) but optimized with list slicing?
        
    # Let's try a different generation strategy for speed:
    sumsA = [0.0] * sizeA
    for i in range(nA):
        val = valsA[i]
        bit = 1 << i
        # Update sums for all masks that include this bit
        # We can iterate over blocks
        for start in range(0, sizeA, bit << 1):
            for mask in range(start + bit, start + (bit << 1)):
                sumsA[mask] = sumsA[mask ^ bit] + val
                
    # Pair sums with masks
    # We want to sort by sum.
    # Create list of (sum, mask)
    pairsA = [(sumsA[m], m) for m in range(sizeA)]
    pairsA.sort(key=lambda x: x[0])
    sorted_sumsA = [p[0] for p in pairsA]
    sorted_masksA = [p[1] for p in pairsA]
    
    if time.time() - start_time > deadline:
        return [0] # Fallback

    # 2. Generate all subset sums for B (as floats)
    sizeB = 1 << nB
    sumsB = [0.0] * sizeB
    for i in range(nB):
        val = valsB[i]
        bit = 1 << i
        for start in range(0, sizeB, bit << 1):
            for mask in range(start + bit, start + (bit << 1)):
                sumsB[mask] = sumsB[mask ^ bit] + val

    if time.time() - start_time > deadline:
        return [0] # Fallback

    # 3. Search for best candidates
    # We want to find maskA, maskB such that sumsA[maskA] + sumsB[maskB] < 1.0
    # and the deficit is minimized.
    
    # Use a heap to keep top K candidates
    import heapq
    K = 500
    candidates = [] # (deficit, maskA, maskB)
    
    for maskB in range(sizeB):
        sumB = sumsB[maskB]
        if sumB >= 1.0:
            continue
        
        target = 1.0 - sumB
        # Find largest sumA < target
        idx = bisect.bisect_left(sorted_sumsA, target)
        if idx > 0:
            maskA = sorted_masksA[idx - 1]
            sumA = sorted_sumsA[idx - 1]
            total = sumA + sumB
            if total < 1.0:
                deficit = 1.0 - total
                # Add to heap
                if len(candidates) < K:
                    heapq.heappush(candidates, (deficit, maskA, maskB))
                else:
                    if deficit < candidates[0][0]:
                        heapq.heapreplace(candidates, (deficit, maskA, maskB))
                        
        if time.time() - start_time > deadline:
            break

    # 4. Verify top candidates with exact Fractions
    best_exact_indices = None
    best_exact_deficit = None
    best_exact_sum = None
    
    # Sort candidates by deficit to check best ones first
    candidates.sort(key=lambda x: x[0])
    
    for deficit, maskA, maskB in candidates:
        if time.time() - start_time > deadline:
            break
            
        # Reconstruct indices
        indices = []
        # A part
        for i in range(nA):
            if maskA & (1 << i):
                indices.append(i)
        # B part
        for i in range(nB):
            if maskB & (1 << i):
                indices.append(nA + i)
        
        if not indices:
            continue
            
        # Calculate exact sum
        s = Fraction(0)
        for idx in indices:
            s += Fraction(1, denoms[idx])
            
        if s < 1:
            exact_deficit = 1 - s
            if best_exact_deficit is None or exact_deficit < best_exact_deficit:
                best_exact_deficit = exact_deficit
                best_exact_sum = s
                best_exact_indices = indices

    # Fallback if no candidate found (should not happen)
    if best_exact_indices is None:
        # Greedy fallback
        selected = []
        total = Fraction(0)
        for i, d in enumerate(denoms):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        best_exact_indices = selected
        best_exact_sum = total
        best_exact_deficit = 1 - total

    # 5. Local Search to improve
    # Try to swap or add elements to reduce deficit further
    current_indices = set(best_exact_indices)
    current_sum = best_exact_sum
    current_deficit = best_exact_deficit
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # Try adding an element
        for add in range(n):
            if add in current_indices:
                continue
            new_sum = current_sum + Fraction(1, denoms[add])
            if new_sum < 1 and new_sum > current_sum:
                current_indices.add(add)
                current_sum = new_sum
                current_deficit = 1 - new_sum
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping one element with another
        # To make this efficient, we can just iterate all pairs. N=47, pairs ~1100.
        # For each pair, check if swap improves sum.
        for rem in list(current_indices):
            for add in range(n):
                if add in current_indices:
                    continue
                new_sum = current_sum - Fraction(1, denoms[rem]) + Fraction(1, denoms[add])
                if new_sum < 1 and new_sum > current_sum:
                    current_indices.discard(rem)
                    current_indices.add(add)
                    current_sum = new_sum
                    current_deficit = 1 - new_sum
                    improved = True
                    break
            if improved:
                break

    return list(current_indices)

if __name__ == "__main__":
    construct()
