import time
import sys

def construct():
    deadline = time.time() + 85.0
    sys.setrecursionlimit(10000)
    
    # Greedy baseline
    def greedy():
        values, products = [], set()
        for x in range(2, 129):
            new_suffixes = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                new_suffixes.append(prod)
            if not products.intersection(new_suffixes):
                values.append(x)
                products.update(new_suffixes)
        return values, products
    
    best_list, _ = greedy()
    best_len = [len(best_list)]
    
    def dfs(values, products, suffixes):
        if time.time() > deadline:
            return
        cur_len = len(values)
        last = values[-1] if values else 1
        if cur_len + (128 - last) <= best_len[0]:
            return
        for x in range(last + 1, 129):
            if time.time() > deadline:
                return
            new_suffixes = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                new_suffixes.append(prod)
            if products.intersection(new_suffixes):
                continue
            new_products = products | set(new_suffixes)
            new_values = values + [x]
            if len(new_values) > best_len[0]:
                best_len[0] = len(new_values)
                best_list[:] = new_values
            dfs(new_values, new_products, new_suffixes)
    
    dfs([], set(), [])
    return best_list
