        current_volts = [random.randint(0, m-1) for _ in range(10)]
        current_cost = get_cost(current_volts)
        
        if current_cost == 0:
            # Found
            ...
            
        T = 10.0
        for step in range(10000):
            if time.time() > deadline - 5: break
            new_volts = current_volts[:]
            i = random.randint(0, 9)
            new_volts[i] = random.randint(0, m-1)
            new_cost = get_cost(new_volts)
            
            if new_cost == 0:
                # Found
                ...
                break
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_volts = new_volts
                current_cost = new_cost
            T *= 0.995
