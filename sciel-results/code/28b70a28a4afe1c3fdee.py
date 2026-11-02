        delta = 0
        for s in block_subsets[i]:
            cov[s] -= 1
            if cov[s] == 0:
                delta += 1
        return delta
