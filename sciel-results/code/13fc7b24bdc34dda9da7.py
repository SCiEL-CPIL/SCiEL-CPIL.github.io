import sys
sys.setrecursionlimit(10000)

def construct():
    # Generate 5-smooth numbers up to 128
    smooth = []
    for a in range(8):
        for b in range(6):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    
    n_smooth = len(smooth)
    best = [None] * 129
    remaining = 128
    current_list = []

    def generate_antichains(k, start_idx, current_sum):
        nonlocal remaining, best, current_list
        
        if len(current_list) == k:
            if 1 <= current_sum <= 128:
                if best[current_sum] is None:
                    best[current_sum] = list(current_list)
                    remaining -= 1
            return
        
        needed = k - len(current_list)
        
        # Pruning: not enough elements left to complete the antichain
        if n_smooth - start_idx < needed:
            return
            
        for i in range(start_idx, n_smooth):
            val = smooth[i]
            
            # Pruning: sum exceeds target max
            if current_sum + val > 128:
                break
                
            # Check antichain property with current_list
            valid = True
            for c in current_list:
                if val % c == 0 or c % val == 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Recurse
            current_list.append(val)
            generate_antichains(k, i + 1, current_sum + val)
            current_list.pop()
            
            if remaining == 0:
                return

    # Iterate k from 1 upwards
    # We start with k=1 manually to handle single smooth numbers efficiently
    # Or just let the function handle k=1.
    # Let's let the function handle all k.
    for k in range(1, 11):
        if remaining == 0:
            break
        generate_antichains(k, 0, 0)

    return best[1:129]
