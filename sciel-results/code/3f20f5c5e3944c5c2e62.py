    def greedy(priority):
        path = [0]
        used_mask = 1 # bit 0
        cur = 0
        while True:
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                # Check if v is adjacent to any previous vertex (except cur)
                # v's neighbors in used_mask should be exactly cur
                if (NEIGHBOR_MASK[v] & used_mask) != (1 << cur):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        return path
    