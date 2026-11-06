        if move_type < 0.4:
            # Swap
            while True:
                x = random.randint(1, N)
                if x in S:
                    break
            while True:
                y = random.randint(1, N)
                if y not in S:
                    break
            # ... rest of swap logic
