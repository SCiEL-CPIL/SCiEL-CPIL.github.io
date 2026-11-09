def _balanced_units(v):
    # v in pool? 
    if v in pool: return pool[v][0]
    bits = [j for j in range(v.bit_length()) if (v>>j)&1]
    if len(bits) == 0: return n
    if len(bits) == 1: return pool[v][0]
    # build balanced tree
    # recursively split bits list in half
    def rec(bitlist):
        if len(bitlist) == 1:
            return (1 << bitlist[0], 0)
        mid = len(bitlist)//2
        va, da = rec(bitlist[:mid])
        vb, db = rec(bitlist[mid:])
        w = add_gate(va, da, vb, db)
        return (w, max(da,db)+1)
    w, d = rec(bits)
    return w
