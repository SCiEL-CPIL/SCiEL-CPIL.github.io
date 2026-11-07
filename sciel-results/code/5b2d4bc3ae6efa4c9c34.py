def construct():
    n = 40
    candidates = list(range(2, n + 1))
    selected = []
    products = {1}
    
    for a in candidates:
        # Calculate new products if a is added
        new_products = set(products)
        for p in products:
            new_products.add(p * a)
            
        if len(new_products) == len(products) * 2:
            # All new products are distinct and distinct from old ones
            selected.append(a)
            products = new_products
            if len(selected) >= 18:
                break
                
    return selected
