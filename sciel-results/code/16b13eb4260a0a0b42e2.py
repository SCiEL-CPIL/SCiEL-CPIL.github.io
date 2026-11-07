    def try_greedy(order):
        current_set = []
        current_products = {1}
        for c in order:
            valid = True
            new_prods = []
            for p in current_products:
                np = p * c
                if np in current_products:
                    valid = False
                    break
                new_prods.append(np)
            if valid:
                current_set.append(c)
                current_products.update(new_prods)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    if best_size == 18:
                        return best_set
        return best_set
