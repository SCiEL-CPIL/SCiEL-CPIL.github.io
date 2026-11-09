<<<<<<< SEARCH
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(bicliques):
            j = i + 1
            while j < len(bicliques):
                L1, R1 = set(bicliques[i][0]), set(bicliques[i][1])
                L2, R2 = set(bicliques[j][0]), set(bicliques[j][1])
                result = can_merge(L1, R1, L2, R2)
                if result is not None:
                    mL, mR = result
                    bicliques[i] = [sorted(int(x) for x in mL), sorted(int(x) for x in mR)]
                    bicliques.pop(j)
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1
=======
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(bicliques):
            j = i + 1
            while j < len(bicliques):
                L1, R1 = set(bicliques[i][0]), set(bicliques[i][1])
                L2, R2 = set(bicliques[j][0]), set(bicliques[j][1])
                result = can_merge(L1, R1, L2, R2)
                if result is not None:
                    mL, mR = result
                    bicliques[i] = [sorted(int(x) for x in mL), sorted(int(x) for x in mR)]
                    bicliques.pop(j)
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1
>>>>>>> REPLACE
