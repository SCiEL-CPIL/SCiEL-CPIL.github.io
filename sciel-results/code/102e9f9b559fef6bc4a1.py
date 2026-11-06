<<<<<<< SEARCH
def _is_known_gate(name, target, qubits):
    """Check if the target is one of the known gates and return exact decomposition."""
    name_lower = name.lower()
    
    if qubits == 2:
        if 'cnot' in name_lower or 'cx' in name_lower:
            # Check if it's CNOT(0,1)
            ref = _build(_CNOT_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _CNOT_2q()
            # Try CNOT(1,0)
            ref = _build([['cx', 1, 0]], 2)
            if _distance(target, ref) < 1e-10:
                return [['cx', 1, 0]]
        
        if 'iswap' in name_lower or 'i-swap' in name_lower:
            ref = _build(_iSWAP_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _iSWAP_2q()
        
        if 'swap' in name_lower:
            ref = _build(_SWAP_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _SWAP_2q()
    
    if qubits == 3:
        if 'toffoli' in name_lower or 'ccnot' in name_lower or 't' == name_lower.strip():
            # Try all 6 permutations of Toffoli
            for c1 in range(3):
                for c2 in range(3):
                    if c2 == c1:
                        continue
                    t = 3 - c1 - c2
                    ref = _build(_toffoli_3q(c1, c2, t), 3)
                    if _distance(target, ref) < 1e-10:
                        return _toffoli_3q(c1, c2, t)
    
    return None
=======
def _fredkin_3q(control, target1, target2):
    """Exact C-SWAP (Fredkin) decomposition with 7 CNOTs."""
    gates = []
    def _t(q): return ['u3', q, math.pi / 4, 0, 0]
    def _tdag(q): return ['u3', q, -math.pi / 4, 0, 0]
    def _cx(c, t): return ['cx', c, t]
    
    # Standard Fredkin(c, t1, t2)
    # 1. CX(t1, t2)
    gates.append(_cx(target1, target2))
    # 2. CX(c, t2)
    gates.append(_cx(control, target2))
    # 3. CX(t1, t2)
    gates.append(_cx(target1, target2))
    # 4. T†(t2) T(t1)
    gates.append(_tdag(target2))
    gates.append(_t(target1))
    # 5. CX(c, t1)
    gates.append(_cx(control, target1))
    # 6. T(t1) T†(t2)
    gates.append(_t(target1))
    gates.append(_tdag(target2))
    # 7. CX(c, t1)
    gates.append(_cx(control, target1))
    # 8. T(t2) T†(t1)
    gates.append(_t(target2))
    gates.append(_tdag(target1))
    # 9. CX(c, t2)
    gates.append(_cx(control, target2))
    # 10. T(t1) T†(t2)
    gates.append(_t(target1))
    gates.append(_tdag(target2))
    # 11. CX(c, t2)
    gates.append(_cx(control, target2))
    # 12. T(t2) T†(t1)
    gates.append(_t(target2))
    gates.append(_tdag(target1))
    
    return gates

def _is_known_gate(name, target, qubits):
    """Check if the target is one of the known gates and return exact decomposition."""
    name_lower = name.lower()
    
    if qubits == 2:
        if 'cnot' in name_lower or 'cx' in name_lower:
            ref = _build(_CNOT_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _CNOT_2q()
            ref = _build([['cx', 1, 0]], 2)
            if _distance(target, ref) < 1e-10:
                return [['cx', 1, 0]]
        
        if 'iswap' in name_lower or 'i-swap' in name_lower:
            ref = _build(_iSWAP_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _iSWAP_2q()
        
        if 'swap' in name_lower:
            ref = _build(_SWAP_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _SWAP_2q()
    
    if qubits == 3:
        # Check Toffoli
        if 'toffoli' in name_lower or 'ccnot' in name_lower or 't' == name_lower.strip():
            for c1 in range(3):
                for c2 in range(3):
                    if c2 == c1:
                        continue
                    t = 3 - c1 - c2
                    ref = _build(_toffoli_3q(c1, c2, t), 3)
                    if _distance(target, ref) < 1e-10:
                        return _toffoli_3q(c1, c2, t)
        
        # Check Fredkin (C-SWAP)
        # Names might be 'fredkin', 'cswap', 'c-swap'
        if 'fredkin' in name_lower or 'cswap' in name_lower or 'c-swap' in name_lower:
            for c in range(3):
                for t1 in range(3):
                    if t1 == c: continue
                    t2 = 3 - c - t1
                    ref = _build(_fredkin_3q(c, t1, t2), 3)
                    if _distance(target, ref) < 1e-10:
                        return _fredkin_3q(c, t1, t2)
    
    return None
>>>>>>> REPLACE

Now replace the optimization part in `synthesize`.

