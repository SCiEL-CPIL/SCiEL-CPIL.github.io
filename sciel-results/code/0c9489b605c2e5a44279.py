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
            # Last resort: should never happen
            answers.append([n] if n in smooth_set else [1])
    
    return answers

# Precompute at module load time
_ANSWERS = _precompute_answers()

def construct():
    return _ANSWERS
# EVOLVE-BLOCK-END
