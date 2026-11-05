# EVOLVE-BLOCK-START
import time

def _precompute_answers():
    """Precompute optimal antichain representations for targets 1..128."""
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    answers = []
    
    for n in range(1, 129):
        # k=1: n itself is smooth
        if n in smooth_set:
            answers.append([n])
            continue
        
        # k=2: find a, b smooth, a+b=n, a<b, b%a!=0 (antichain)
        found = None
        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set and b % a != 0:
                found = [a, b]
                break
        
        if found is not None:
            answers.append(found)
            continue
        
        # k=3 fallback (should not be needed for n<=128, but safety)
        found = None
        for i in range(len(smooth)):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, len(smooth)):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    if b % a != 0 and c % a != 0 and c % b != 0:
                        found = [a, b, c]
                        break
            if found:
                break
        if found:
            answers.append(found)
        else:
            # Last resort: should never happen for n<=128
            # If it does, it's an error, but we must return something.
            # Returning [n] is invalid if n is not smooth and sum != n (wait, [n] sums to n).
            # But [n] is not an antichain if n is not smooth? No, [n] is a single term.
            # Antichain condition: "no chosen term may divide another".
            # With 1 term, there are no pairs, so it's always an antichain.
            # BUT the term must be smooth.
            # If n is not smooth, [n] is invalid.
            # The fallback [1] is smooth, but sum is 1 != n. Invalid.
            # This branch should never be reached.
            raise ValueError(f"No representation found for {n}")
    
    return answers

# Precompute at module load time
_ANSWERS = _precompute_answers()

def construct():
    return _ANSWERS
# EVOLVE-BLOCK-END
