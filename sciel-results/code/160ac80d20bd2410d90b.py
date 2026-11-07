    def check_add(products, x):
        new_prods = []
        for p in products:
            np = p * x
            if np in products:
                return False, None
            new_prods.append(np)
        # No need to check new_prods for internal duplicates if products are distinct and x is constant
        # Because if p1*x == p2*x, then p1==p2.
        return True, products.union(new_prods)
