            if stuck_counter % 1000 == 0:
                # Kick: flip 10 random edges
                for _ in range(10):
                    colors[random.randint(0, total_triples - 1)] = 1 - colors[random.randint(0, 1)]
