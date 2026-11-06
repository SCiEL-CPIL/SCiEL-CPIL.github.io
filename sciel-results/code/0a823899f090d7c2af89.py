# Global cache
_INITIAL_WIRES = None
_MASKS = None

def _get_initial_wires():
    global _INITIAL_WIRES
    if _INITIAL_WIRES is None:
        n = N
        wires = [0] * n
        for i in range(n):
            mask = 0
            for k in range(1 << n):
                if (k >> i) & 1:
                    mask |= (1 << k)
            wires[i] = mask
        _INITIAL_WIRES = wires
    return _INITIAL_WIRES[:]

def _get_masks():
    global _MASKS
    if _MASKS is None:
        n = N
        masks = [0] * n
        for k in range(1 << n):
            pop = bin(k).count('1')
            bit = 1 << k
            for i in range(n):
                if pop >= n - i:
                    masks[i] |= bit
        _MASKS = masks
    return _MASKS

def _verify(network):
    n = N
    wires = _get_initial_wires()
    masks = _get_masks()
    
    for i, j in network:
        new_i = wires[i] & wires[j]
        new_j = wires[i] | wires[j]
        wires[i] = new_i
        wires[j] = new_j
        
    for i in range(n):
        if wires[i] != masks[i]:
            return False
    return True
