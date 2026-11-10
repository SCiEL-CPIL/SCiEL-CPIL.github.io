    best_greedy_pairs = []
    best_greedy_size = 0
    
    for _ in range(10):
        if time.time() > DEADLINE - 5: break
        g_pairs = []
        mods = moduli[:]
        random.shuffle(mods)
        for m in mods:
            m_idx = mod_to_idx[m]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2) in g_pairs:
                    m2_idx = mod_to_idx[m2]
                    g = gcd_table[m_idx][m2_idx]
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    g_pairs.append([r, m])
                    break
        if len(g_pairs) > best_greedy_size:
            best_greedy_size = len(g_pairs)
            best_greedy_pairs = [p[:] for p in g_pairs]
            
    best_pairs = best_greedy_pairs
    best_size = best_greedy_size
