                old_total = count0 + count1
                # ... flip ...
                new_total = count0 + count1
                delta = new_total - old_total
                if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                    # accept
                    pass
                else:
                    # revert
                    # flip back
                    # ...
