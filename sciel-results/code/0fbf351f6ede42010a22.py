                y = random.choice(list(selected))
                selected.remove(y)
                update_counts(y, -1, selected)
