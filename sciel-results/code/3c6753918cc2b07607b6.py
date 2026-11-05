import time

def construct():
    P = 101
    DENOMS = list(range(1, 11))
    inv = [0] * 11
    for x in DENOMS:
        inv[x] = pow(x, -1, P)
    
    # best_size[r] = minimum subset size achieving residue r
    # best_mask[r] = corresponding bitmask
    best_size = [10**9] * P
    best_mask = [0] * P
    
    # Enumerate all non-empty subsets
    # Use bit manipulation for speed
    for mask in range(1, 1 << 10):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += inv[idx + 1]  # idx+1 because denominators are 1..10
            m >>= 1
            idx += 1
        s %= P
        sz = bin(mask).count('1')
        if sz < best_size[s]:
            best_size[s] = sz
            best_mask[s] = mask
    
    # Convert masks back to lists of denominators
    result = []
    for r in range(P):
        if best_mask[r] != 0:
            terms = []
            for x in range(1, 11):
                if best_mask[r] & (1 << (x - 1)):
                    terms.append(x)
            result.append(terms)
        else:
            # Should not happen, but provide a valid non-empty list
            result.append([1])
    
    return result
