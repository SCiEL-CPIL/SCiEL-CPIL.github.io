<<<<<<< SEARCH
    def crossover(parent1, parent2):
        """Uniform crossover between two sets."""
        p1 = list(parent1)
        p2 = list(parent2)
        child = set()
        for i in range(32):
            if random.random() < 0.5:
                child.add(p1[i])
            else:
                child.add(p2[i])
        # Ensure exactly 32 distinct elements
        if len(child) != 32:
            # Fix by adding/removing
            child_list = list(child)
            if len(child_list) > 32:
                # Remove excess
                excess = random.sample(child_list, len(child_list) - 32)
                for e in excess:
                    child.discard(e)
            while len(child) < 32:
                val = random.randint(1, 512)
                if val not in child:
                    child.add(val)
        return child
=======
    def crossover(parent1, parent2):
        """Uniform crossover between two sets."""
        p1 = list(parent1)
        p2 = list(parent2)
        child = set()
        for i in range(32):
            if random.random() < 0.5:
                child.add(p1[i])
            else:
                child.add(p2[i])
        # Ensure exactly 32 distinct elements
        if len(child) != 32:
            # Fix by adding/removing
            child_list = list(child)
            if len(child_list) > 32:
                # Remove excess
                excess = random.sample(child_list, len(child_list) - 32)
                for e in excess:
                    child.discard(e)
            while len(child) < 32:
                val = random.randint(1, 512)
                if val not in child:
                    child.add(val)
        return child
    
    def smart_mutate(individual):
        """Mutation that tries to add elements creating new progressions."""
        ind = set(individual)
        ind_list = sorted(ind)
        ind_set = set(ind)
        
        # Find potential new elements that could create progressions
        candidates = set()
        for i in range(len(ind_list)):
            a = ind_list[i]
            for j in range(i + 1, len(ind_list)):
                b = ind_list[j]
                d = b - a
                c1 = a + 2 * d
                if 1 <= c1 <= 512 and c1 not in ind_set:
                    candidates.add(c1)
                c2 = b + d
                if 1 <= c2 <= 512 and c2 not in ind_set:
                    candidates.add(c2)
                c3 = a - d
                if 1 <= c3 <= 512 and c3 not in ind_set:
                    candidates.add(c3)
        
        if candidates:
            # Pick a random candidate to add
            to_add = random.choice(list(candidates))
            to_remove = random.choice(list(ind))
            ind.discard(to_remove)
            ind.add(to_add)
        else:
            # Fallback random mutation
            to_remove = random.choice(list(ind))
            ind.discard(to_remove)
            while True:
                to_add = random.randint(1, 512)
                if to_add not in ind:
                    ind.add(to_add)
                    break
        
        return ind
>>>>>>> REPLACE

