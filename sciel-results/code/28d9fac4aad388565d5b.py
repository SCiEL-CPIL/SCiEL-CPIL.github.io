"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # All odd numbers form a valid set: for distinct odd a,b, a+b is even,
    # a*b is odd, so (a+b) cannot divide (a*b).
    odds = list(range(1, 257, 2))
    
    # Find even numbers e such that for all odd a in [1,255], (a+e) does not divide e*a.
    # (a+e) | e*a  <=>  (a+e) | e^2  (since e*a - e*(a+e) = -e^2)
    # So e is compatible with all odds iff no odd d in {e+1, e+3, ..., e+255} divides e^2.
    valid_evens = []
    for e in range(2, 257, 2):
        e2 = e * e
        compatible = True
        for a in range(1, 256, 2):
            s = a + e
            if s <= e2 and e2 % s == 0:
                compatible = False
                break
        if compatible:
            valid_evens.append(e)
    
    # Greedily select a subset of valid_evens with no bad even-even pairs.
    # (e1+e2) | e1*e2  <=>  (e1+e2) | e1^2
    kept_evens = []
    for e in valid_evens:
        ok = True
        for e2 in kept_evens:
            s = e + e2
            if s <= e * e and (e * e) % s == 0:
                ok = False
                break
        if ok:
            kept_evens.append(e)
    
    return odds + kept_evens
# EVOLVE-BLOCK-END
