    def random_greedy():
        elems = list(all_elems)
        random.shuffle(elems)
        seq = []
        prods = set()
        for x in elems:
            # Check if x can be added
            # We need to check all possible positions? 
            # Or just append?
            # Appending is faster. Let's just append.
            # But inserting in the middle might be better.
            # For simplicity, let's just append.
            new_prods = []
            p = x
            for a in reversed(seq):
                new_prods.append(p)
                p *= a
            new_prods.append(p)
            
            if not prods.intersection(new_prods):
                seq.append(x)
                prods.update(new_prods)
        return seq

    # Try a few random greedies
    best_init = []
    for _ in range(10):
        s = random_greedy()
        if len(s) > len(best_init):
            best_init = s
