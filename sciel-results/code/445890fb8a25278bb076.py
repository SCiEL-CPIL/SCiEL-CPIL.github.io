            if random.random() < 0.5:
                # Small move
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
            elif random.random() < 0.5:
                # Medium move
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
            else:
                # Large move
                dx = random.randint(-100, 100)
                dy = random.randint(-100, 100)
