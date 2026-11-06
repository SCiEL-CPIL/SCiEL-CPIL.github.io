<<<<<<< SEARCH
            pattern = []
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                r = random.random()
                if r < 0.3:
                    pattern.append([])
                elif r < 0.7:
                    pattern.append([random.choice([a, b, c])])
                else:
                    pattern.append([random.sample([a, b, c], 2)])
=======
            pattern = []
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                r = random.random()
                if r < 0.3:
                    pattern.append([])
                elif r < 0.7:
                    pattern.append([random.choice([a, b, c])])
                else:
                    pattern.append(random.sample([a, b, c], 2))
>>>>>>> REPLACE
