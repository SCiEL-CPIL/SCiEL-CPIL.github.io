<<<<<<< SEARCH
        # Multiple greedy starts
        orders = [
            sorted(range(n), key=lambda i: -abs(arr[i])),
            sorted(range(n), key=lambda i: abs(arr[i])),
            list(range(n))
        ]
        rng.shuffle(orders[2])
        
        best_included = set()
        for order in orders:
            inc = greedy_init(order)
            if len(inc) > len(best_included):
                best_included = inc
=======
        # Multiple greedy starts with diverse orderings
        orders = [
            sorted(range(n), key=lambda i: -abs(arr[i])),
            sorted(range(n), key=lambda i: abs(arr[i])),
            sorted(range(n), key=lambda i: arr[i]),
            sorted(range(n), key=lambda i: -arr[i]),
            list(range(n))
        ]
        for _ in range(3):
            orders.append(list(range(n)))
        for order in orders:
            rng.shuffle(order)
        
        best_included = set()
        for order in orders:
            inc = greedy_init(order)
            if len(inc) > len(best_included):
                best_included = inc
>>>>>>> REPLACE

