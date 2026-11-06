    def remove_set(idx):
        # Find position
        pos = family.index(idx)
        family[pos] = family[-1]
        family.pop()
        fam_set.remove(idx)
        # Update counts
        ...
    